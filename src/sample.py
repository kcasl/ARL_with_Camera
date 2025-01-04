import numpy as np
import onnxruntime as ort

# ONNX 모델 로드
model_path = '../model/KSEFAdversarialCar.onnx'  # 모델 파일 경로
session = ort.InferenceSession(model_path)

# 모델의 입력 정보 확인
input_details = session.get_inputs()
output_details = session.get_outputs()

# 필요한 입력값 생성
input_feed = {}
for input_detail in input_details:
    input_name = input_detail.name
    input_shape = input_detail.shape

    # 형상 값 정리
    processed_shape = []
    for dim in input_shape:
        if isinstance(dim, str) or dim is None:  # 문자열 또는 None 확인
            processed_shape.append(1)  # 기본 값을 1로 설정
        else:
            processed_shape.append(dim)

    # 랜덤 입력값 생성
    input_feed[input_name] = np.random.rand(*processed_shape).astype(np.float32)

# 모델 실행
output = session.run(None, input_feed)

# 출력 결과
output_names = [output_detail.name for output_detail in output_details]
output_data = dict(zip(output_names, output))

print("Output Data:", output_data)
