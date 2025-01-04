import onnxruntime as ort
import numpy as np

model_paths = [
    "ERL.onnx",
    "ERL.onnx",
    "Rain_ERL.onnx",
    "Sandstorm_ERL.onnx"
]

sessions = [ort.InferenceSession(model_path) for model_path in model_paths]

# obs_0: (n, h:84, w:84, c:6)
obs_0 = np.random.rand(1, 84, 84, 6).astype(np.float32)

# obs_1: (n, h:1, w:1, c:3)
obs_1 = np.random.rand(1, 1, 1, 3).astype(np.float32)

# action_masks: (n, h:1, w:1, c:4)
action_masks = np.random.rand(1, 1, 1, 4).astype(np.float32)

outputs = []
for session in sessions:
    input_names = [inp.name for inp in session.get_inputs()]

    # ONNX 모델 실행
    output = session.run(
        None,  # 모든 출력값 가져오기
        {
            input_names[0]: obs_0,
            input_names[1]: obs_1,
            input_names[2]: action_masks
        }
    )
    outputs.append(output)

discrete_actions_idx = 2  # Outputs에서 discrete_actions의 인덱스
aggregated_output = np.mean(
    [output[discrete_actions_idx] for output in outputs], axis=0
)

policy_decision = np.argmax(aggregated_output)

print("Aggregated Output (discrete_actions):", aggregated_output)
print("Policy Decision (argmax):", policy_decision)
