from typing import List, Dict
from pathlib import Path
import openai
import cv2


def inference_with_openai_api(
    messages,
    model_id="/bigwork/nhwpduep/master_thesis/models/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
):
    openai.api_key = "..."
    vllm_host = "http://0.0.0.0:8000"
    openai.api_base = f"{vllm_host}/v1"

    response = openai.ChatCompletion.create(
        model=model_id,
        messages=messages,
        n=1,
    )
    print(response)
    return response["choices"][0].message.content  # type: ignore


def prepare_video_message(frame_dir: Path, fps: int) -> List[Dict[str, str]]:
    frames = frame_dir.glob("*.jpg")
    video_message = []
    i = -1
    for i, frame in enumerate(frames):
        video_message.append({"type": "text", "text": f"<{i/fps:.2f} seconds>"})
        video_message.append(
            {
                "type": "image_url",
                "image_url": {"url": f"file://{str(frame.absolute())}"},
            }
        )
    print(f"Number of frames: {i+1}")
    return video_message


def extract_frames(video_path: Path, frame_dir: Path, fps: int):
    frame_dir.mkdir(exist_ok=True)
    old_frames = frame_dir.glob("*.jpg")
    for frame in old_frames:
        frame.unlink()

    video = cv2.VideoCapture(video_path)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    step = original_fps // fps
    print(f"{original_fps=}, {fps=}, {step=}")
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if i % step == 0:
            cv2.imwrite(frame_dir / f"{video_path.stem}-{i:04d}.jpg", frame)
        i += 1

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video = Path(
        "/bigwork/nhwpduep/master_thesis/dr-eureka-go2/runs/eureka/2026-02-21_09:35:30_GL_Go2_Qwen-30BQ-nt/1/14_2026-02-22_10:16:49/videos/final.mp4"
    )
    frame_dir = Path(
        "/bigwork/nhwpduep/master_thesis/dr-eureka-go2/examples/video_frames"
    )
    fps = 25
    extract_frames(video, frame_dir, fps)

    text = """
        You are a reward engineer trying to give precise and helpful feedback to your colleague.  
        The following video shows multiple rollouts of a quadruped robot balancing on a ball. The ball does only move due to actions of the robot. The robot's policy shall be deployed safely in the real world. Therefore, the robot should have a stable position on top of the ball and make only minimal smooth actions to stay in balance. Analyze the recording as follows:  
            (1) Describe the movements of the robot sequentially and in detail. Especially look into the leg and foot positions and dynamics.  
            (2) Pick movements that are particularly beneficial for maintaining balance on the ball. Describe foot/leg positions with respect to the robots body and the ball and the dynamics of the movement.
            (3) Pick movements that are particularly detrimental for maintaining balance on the ball. Describe foot/leg positions with respect to the robots body and the ball and the dynamics of the movement.
            (4) Summarize your findings.  
        Mark each section with the index: <**index**>
    """
    print(f"{text=}")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
                *prepare_video_message(frame_dir, fps),
            ],
        }
    ]
    print(f"{messages=}")
    response = inference_with_openai_api(messages)
    print(f"{response=}")
    print(f"{response.split('</think>')[-1]=}")
