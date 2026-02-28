import argparse
import random
from typing import Tuple

import torch

from hybrid_model import HybridDQN
from rl_env import VanetRLEnv


def evaluate_dqn(
    model_path: str = "fixed_dqn.pth",
    gnn_weights_path: str = "pretrained_sybil_gnn.pth",
    csv_file: str = "small_gnn_dqn_data_replay_dataset",
    num_episodes: int = 1,
) -> Tuple[float, float]:
    """
    Evaluate a trained HybridDQN on the VANET RL environment.

    - Loads the GNN backbone from `gnn_weights_path`.
    - Loads the trained DQN (and GNN) weights from `model_path`.
    - Runs greedy policy (no exploration) over `num_episodes`.
    - Uses the same environment and action semantics as training:
        action 0 = Accept (Normal), action 1 = Reject (Attack).

    Returns
    -------
    avg_reward_per_step : float
    accuracy            : float  (vs. `is_attack` labels for chosen targets)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating DQN on {device}")

    # Environment uses the same dataset format as training
    env = VanetRLEnv(csv_file)

    policy_net = HybridDQN(gnn_weights_path=gnn_weights_path).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    total_reward = 0.0
    total_steps = 0

    correct = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for episode in range(num_episodes):
            print(f"\nStarting evaluation episode {episode + 1}/{num_episodes}")
            state = env.reset()
            done = False
            graph_count = 0

            while not done:
                num_senders = state.x.shape[0] - 1
                if num_senders < 1:
                    state, _, done = env.step(0, 0)
                    continue

                graph_count += 1

                # EVALUATE ALL SENDERS
                for target_idx in range(1, num_senders + 1):
                    batch = torch.zeros(state.x.shape[0], dtype=torch.long, device=device)
                    q_values = policy_net(
                        state.x.to(device),
                        state.edge_index.to(device),
                        batch,
                        target_idx,
                    )

                    action = int(q_values.argmax(dim=1).item())
                    true_label = int(state.y[target_idx].item())
                    pred_attack = 1 if action == 1 else 0

                    if pred_attack == true_label:
                        correct += 1
                    if pred_attack == 1 and true_label == 1:
                        tp += 1
                    elif pred_attack == 1 and true_label == 0:
                        fp += 1
                    elif pred_attack == 0 and true_label == 0:
                        tn += 1
                    elif pred_attack == 0 and true_label == 1:
                        fn += 1

                    total_steps += 1

                next_state, reward, done = env.step(0, 1)
                total_reward += float(reward)
                state = next_state if next_state is not None else state

            print(f"Evaluated {graph_count} graph snapshots, {total_steps} total targets")

    if total_steps == 0:
        print("No steps were taken during evaluation.")
        return 0.0, 0.0

    avg_reward_per_step = total_reward / graph_count
    accuracy = correct / total_steps

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mdr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    print("\n=== Evaluation Results ===")
    print(f"Graphs evaluated         : {graph_count}")
    print(f"Total targets evaluated  : {total_steps}")
    print(f"Average reward per step  : {avg_reward_per_step:.4f}")
    print(f"Accuracy                 : {accuracy:.4f}")
    print(f"TP / FP / TN / FN        : {tp} / {fp} / {tn} / {fn}")
    print(f"Precision (attack)       : {precision:.4f}")
    print(f"Recall (attack)          : {recall:.4f}")
    print(f"F1-score (attack)        : {f1:.4f}")
    print(f"False Positive Rate      : {fpr:.4f}")
    print(f"Miss Detection Rate      : {mdr:.4f}")

    return avg_reward_per_step, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HybridDQN on the VANET RL environment."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_hybrid_dqn.pth",
        help="Path to trained HybridDQN weights.",
    )
    parser.add_argument(
        "--gnn-weights-path",
        type=str,
        default="pretrained_sybil_gnn.pth",
        help="Path to pre-trained SybilGNN weights used inside HybridDQN.",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="small_gnn_dqn_dataset.csv",
        help="CSV file used by VanetRLEnv (same format as training).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes (passes over the dataset).",
    )

    args = parser.parse_args()

    evaluate_dqn(
        model_path=args.model_path,
        gnn_weights_path=args.gnn_weights_path,
        csv_file=args.csv_file,
        num_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()

