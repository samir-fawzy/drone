from __future__ import annotations

import asyncio
import random
import math
import logging
import time  # تم الإضافة لحساب الزمن الفعلي
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Tuple

from controllers.drone_controller import DroneController
from intelligence.rl_agent import RLAgent
from intelligence.path_planner import PathPlanner, FlightMode
from services.vision_service import VisionService
from utils.logger import setup_logger


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Engineering
# ═══════════════════════════════════════════════════════════════════════════════
def compute_reward(
    current_pose: Dict[str, float],
    previous_pose: Dict[str, float],
    target_pose: Dict[str, float],
    collision: bool,
    step: int,
    max_steps: int = 500,
) -> Tuple[float, bool]:
    """
    Returns (reward, done).
    """
    prev_dist = math.sqrt(
        (previous_pose["x"] - target_pose["x"]) ** 2 +
        (previous_pose["y"] - target_pose["y"]) ** 2
    )
    curr_dist = math.sqrt(
        (current_pose["x"] - target_pose["x"]) ** 2 +
        (current_pose["y"] - target_pose["y"]) ** 2
    )

    # Distance improvement (positive if getting closer)
    reward = (prev_dist - curr_dist) * 2.0
    # Time penalty
    reward -= 0.05
    # Altitude penalty (keep near target z)
    reward -= abs(current_pose.get("z", 0) - target_pose.get("z", 0)) * 0.01

    # Success
    if curr_dist < 3.0:
        reward += 100.0
        return reward, True

    # Collision
    if collision:
        reward -= 50.0
        return reward, True

    # Out of bounds
    if abs(current_pose["x"]) > 200 or abs(current_pose["y"]) > 200:
        reward -= 20.0
        return reward, True

    # Timeout
    if step >= max_steps:
        reward -= 10.0
        return reward, True

    return reward, False


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════
async def train(
    num_episodes: int = 200,
    max_steps: int = 500,
    save_every: int = 10,
):
    project_root = Path(__file__).resolve().parent
    logger = setup_logger(log_dir=project_root / "data" / "logs" / "training")
    logger.info("🎓 Starting RL Training in AirSim")

    controller = DroneController(logger=logger)
    vision = VisionService(project_root=project_root, logger=logger)
    planner = PathPlanner(logger=logger, mode=FlightMode.NORMAL)
    agent = RLAgent(
        logger=logger,
        model_path=project_root / "models" / "dqn_drone_model.pth",
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    )

    await controller.connect()

    # استخدام Try-Except-Finally لحماية التدريب وضمان حفظ الموديل
    try:
        for episode in range(num_episodes):
            logger.info("=== Episode %s / %s | ε=%.3f ===", episode + 1, num_episodes, agent.epsilon)

            # Random target for generalization
            target_pose = {
                "x": random.uniform(-40.0, 40.0),
                "y": random.uniform(-40.0, 40.0),
                "z": random.uniform(-15.0, -5.0),
            }

            await controller.takeoff()
            await asyncio.sleep(2.0)  # Stabilize at takeoff

            current_pose = await controller.get_pose()
            frame = await vision.get_frame()
            detections = await vision.detect_objects(frame)

            state: Dict[str, Any] = {
                "pose": current_pose,
                "target": target_pose,
                "detections": detections,
                "step": 0,
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            }

            previous_pose = current_pose
            episode_reward = 0.0
            step = 0
            
            # تسجيل الوقت لحساب السرعة بدقة
            last_time = time.time()

            while step < max_steps:
                # --- THINK -------------------------------------------------------
                action = await agent.select_action(state)

                # --- ACT ---------------------------------------------------------
                movement_cmd = planner.plan_next_move(
                    detections=state["detections"],
                    rl_action=action,
                )
                
                # تنفيذ الحركة (يُفضل أن تكون الدالة في الكنترولر تستخدم waitByVelocity مثلاً لتزامن أفضل)
                await controller.move(**movement_cmd)

                # --- SENSE -------------------------------------------------------
                next_pose = await controller.get_pose()
                current_time = time.time()
                
                # حساب الفارق الزمني (dt) لتجنب القسمة على صفر
                dt = current_time - last_time
                if dt <= 0:
                    dt = 0.001
                
                frame = await vision.get_frame()
                next_detections = await vision.detect_objects(frame)

                # تحديث حالة الاصطدام برمجياً (يجب أن يحتوي DroneController على دالة لجلب هذه المعلومة من AirSim)
                collision = await controller.has_collided()

                # حساب السرعة الفعلية بناءً على المسافة المقطوعة والزمن الفعلي
                next_state: Dict[str, Any] = {
                    "pose": next_pose,
                    "target": target_pose,
                    "detections": next_detections,
                    "step": step + 1,
                    "velocity": {
                        "x": (next_pose["x"] - previous_pose["x"]) / dt,
                        "y": (next_pose["y"] - previous_pose["y"]) / dt,
                        "z": (next_pose["z"] - previous_pose["z"]) / dt,
                    },
                }

                reward, done = compute_reward(next_pose, previous_pose, target_pose, collision, step, max_steps)
                episode_reward += reward

                # --- LEARN -------------------------------------------------------
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train_step()

                if loss is not None and step % 50 == 0:
                    logger.debug("Step %s | Loss: %.4f | Reward: %.2f", step, loss, reward)

                # Update step variables
                state = next_state
                previous_pose = next_pose
                last_time = current_time
                step += 1

                if done:
                    break

            # Episode cleanup
            await controller.land()
            agent.update_epsilon()

            logger.info(
                "Episode %s finished | Steps: %s | Total Reward: %.2f | Final Dist: %.2f",
                episode + 1, step, episode_reward,
                math.sqrt((previous_pose["x"] - target_pose["x"])**2 + (previous_pose["y"] - target_pose["y"])**2)
            )

            # الحفظ المرحلي لضمان عدم ضياع العمل
            if (episode + 1) % save_every == 0:
                agent.save()
                logger.info("Checkpoint saved.")

    except KeyboardInterrupt:
        logger.warning("⚠️ التدريب تم إيقافه يدوياً من قبل المستخدم (KeyboardInterrupt)!")
    except Exception as e:
        logger.error("❌ حدث خطأ فادح أدى لإيقاف التدريب: %s", str(e), exc_info=True)
    finally:
        # هذه الكتلة ستعمل دائماً سواء انتهى التدريب بنجاح، تم إيقافه، أو حدث خطأ
        logger.info("💾 جاري حفظ نموذج التدريب وإغلاق الاتصالات...")
        agent.save()
        await controller.disconnect()
        vision.shutdown()
        logger.info("🏁 تمت عملية الإنهاء بأمان.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asyncio.run(train())