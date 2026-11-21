"""Local photo segmentation for WalkScapes."""

import argparse
import base64
import math
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request
from PIL import Image, UnidentifiedImageError

from live_overlay import (
    indoor_input, indoor_regions, render_regions,
)
from outdoor_channel import channel_input, render_channel
from walking_guidance import render_guidance
from walkscapes import COLORS, PROJECT, SIZE, overlay, predict_mask


PAGE = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WalkScapes 图像与视频分割</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; color: #182924; background: #f5f7f6; font: 16px/1.5 "Microsoft YaHei UI", system-ui, sans-serif; }
  main { max-width: 780px; margin: 6vh auto; padding: 0 20px; }
  h1 { margin: 0 0 22px; font-size: clamp(24px, 5vw, 34px); }
  form { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 16px; background: white; border: 1px solid #dce6e1; border-radius: 12px; }
  form + form { margin-top: 12px; }
  input { flex: 1; min-width: min(100%, 240px); font: inherit; }
  button, .download { border: 0; border-radius: 8px; padding: 9px 15px; font: inherit; cursor: pointer; }
  button { color: white; background: #12614d; }
  button:disabled { opacity: .7; cursor: wait; }
  .error { color: #9a2b24; }
  .results { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin: 20px 0 12px; }
  figure { margin: 0; }
  figcaption { margin-bottom: 6px; }
  img { display: block; width: 100%; border-radius: 10px; }
  video { display: block; max-width: 100%; max-height: 72vh; margin: 12px 0; border-radius: 10px; }
  .live-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
  #live-views video, #live-views canvas { display: block; width: 100%; margin: 0; border-radius: 10px; background: #15211e; }
  [hidden] { display: none !important; }
  @media (max-width: 540px) { #live-views { grid-template-columns: 1fr; } }
  .legend { display: flex; flex-wrap: wrap; gap: 8px 14px; padding: 0; list-style: none; font-size: 14px; }
  .legend span { display: inline-block; width: .8em; height: .8em; border-radius: 2px; margin-right: 5px; }
  .download { display: inline-block; color: #125442; background: #e4f1eb; text-decoration: none; }
</style>
<main>
  <h1>WalkScapes 分割</h1>
  <section id="live" data-frame-url="{{ url_for('live_frame') }}">
    <h2>实时视频</h2>
    <div class="live-controls">
      <label>场景 <select id="live-mode">
        <option value="outdoor">室外通道</option>
        <option value="indoor">室内走廊</option>
      </select></label>
      <label id="live-enhance-control" hidden><input id="live-enhance" type="checkbox">低光增强</label>
      <button id="start-camera" type="button">开启摄像头</button>
      <label>本地视频 <input id="live-file" type="file" accept="video/*"></label>
      <button id="stop-live" type="button" disabled>停止</button>
    </div>
    <p id="live-status" role="status"></p>
    <div id="live-views" class="results" hidden>
      <figure><figcaption>原画面</figcaption><video id="live-source" playsinline muted></video></figure>
      <figure><figcaption>实时分割</figcaption><canvas id="live-overlay" aria-label="实时分割画面"></canvas></figure>
    </div>
    <p>白色箭线为画面内步行方向参考，绿色为候选通道，橙色为障碍，蓝色为室内台阶。实际通行需结合交通、距离与地面高差判断。</p>
  </section>
  <form id="photo-form" method="post" enctype="multipart/form-data">
    <input name="photo" type="file" accept="image/jpeg,image/png,image/webp" aria-label="选择照片" required>
    <button type="submit">开始分割</button>
  </form>
  <form id="video-form" action="{{ url_for('video') }}" method="post" enctype="multipart/form-data">
    <input name="video" type="file" accept=".mp4,.mov,.avi,.webm" aria-label="选择视频" required>
    <button type="submit">导出分割视频</button>
  </form>
  <p id="video-status" role="status"></p>
  <div id="video-result" hidden>
    <video controls playsinline aria-label="视频分割结果"></video>
    <a class="download" download="walkscapes-video.webm">下载分割视频</a>
  </div>
  {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
  {% if result %}
  <div class="results">
    <figure><figcaption>原图</figcaption><img src="{{ result.original }}" alt="上传照片"></figure>
    <figure><figcaption>分割叠图</figcaption><img src="{{ result.overlay }}" alt="彩色分割结果"></figure>
  </div>
  <ul class="legend" aria-label="分割类别">
  {% for label, color in result.legend %}<li><span style="background: {{ color }}"></span>{{ label }}</li>{% endfor %}
  </ul>
  <a class="download" href="{{ result.mask }}" download="walkscapes-mask.png">下载类别掩码</a>
  {% endif %}
</main>
<script>
  const form = document.querySelector('#photo-form');
  const button = form.querySelector('button');
  form.addEventListener('submit', () => {
    button.disabled = true;
    button.textContent = '分割中…';
  });
  window.addEventListener('pageshow', () => {
    button.disabled = false;
    button.textContent = '开始分割';
  });
  const videoForm = document.querySelector('#video-form');
  const status = document.querySelector('#video-status');
  const videoResult = document.querySelector('#video-result');
  const player = videoResult.querySelector('video');
  const download = videoResult.querySelector('a');
  let videoUrl;
  videoForm.addEventListener('submit', async event => {
    event.preventDefault();
    const submit = videoForm.querySelector('button');
    submit.disabled = true;
    videoResult.hidden = true;
    player.pause();
    player.removeAttribute('src');
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    status.textContent = '正在逐帧分割，视频越长耗时越久…';
    try {
      const endpoint = new URL(videoForm.action, location.href);
      endpoint.searchParams.set('mode', document.querySelector('#live-mode').value);
      const response = await fetch(endpoint, {method: 'POST', body: new FormData(videoForm)});
      if (!response.ok) throw new Error(await response.text());
      videoUrl = URL.createObjectURL(await response.blob());
      player.src = videoUrl;
      download.href = videoUrl;
      videoResult.hidden = false;
      status.textContent = '视频分割完成';
    } catch (error) {
      status.textContent = error.message || '视频处理失败，请重试。';
    } finally {
      submit.disabled = false;
    }
  });
</script>
<script src="{{ url_for('static', filename='live.js') }}" defer></script>
</html>"""

CLASSES_ZH = (
    "道路", "人行道", "墙体", "障碍物", "地面", "天空", "行人",
    "车辆", "植被", "自行车道", "斑马线", "动物", "火车", "水面",
)


def image_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def segment_video(source, destination, infer, render_frame=None) -> None:
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, frame = capture.read()
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError("无法读取视频，请换一个 MP4、MOV、AVI 或 WebM 文件。")
        height, width = frame.shape[:2]
        scale = min(1, 960 / max(width, height))
        size = tuple(max(2, int(dimension * scale) // 2 * 2) for dimension in (width, height))
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
        if not writer.isOpened():
            raise RuntimeError("无法启动 WebM 视频编码。")
        while ok:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if render_frame is None:
                mask = infer(image.resize(SIZE, Image.Resampling.BILINEAR))
                labels = Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
                result = overlay(image.resize(size, Image.Resampling.BILINEAR), np.asarray(labels))
            else:
                result = render_frame(image.resize(size, Image.Resampling.BILINEAR))
            writer.write(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR))
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def create_app() -> Flask:
    import torch
    from transformers import SegformerForSemanticSegmentation

    cv2.setNumThreads(1)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(PROJECT / "model").to(device).eval()
    outdoor_model = SegformerForSemanticSegmentation.from_pretrained(
        PROJECT / "model-outdoor", local_files_only=True,
    ).to(device).eval()
    indoor_path = PROJECT / "model-indoor"
    indoor_model = (
        SegformerForSemanticSegmentation.from_pretrained(indoor_path, local_files_only=True)
        .to(device).eval() if indoor_path.is_dir() else None
    )
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="walkscapes")
    inference_slot = Lock()

    def infer(image):
        with inference_slot:
            return worker.submit(predict_mask, image, model, device).result()

    def live_result(image, indoor, enhance):
        if indoor:
            frame, clipped = indoor_input(image, enhance)
            labels, confidence = predict_mask(frame, indoor_model, device, with_confidence=True)
            regions = indoor_regions(labels, confidence, clipped)
            return render_guidance(render_regions(image, *regions), regions[0])
        labels, confidence = predict_mask(
            channel_input(image), outdoor_model, device, with_confidence=True,
        )
        return render_channel(image, labels, confidence, with_guidance=True)

    if device.type == "cuda":
        for size in ((960, 540), (540, 960)):
            sample = Image.new("RGB", size)
            worker.submit(live_result, sample, False, False).result()
            if indoor_model is not None:
                worker.submit(live_result, sample, True, False).result()

    @app.post("/live-frame")
    def live_frame():
        started = perf_counter()
        request.max_content_length = 1024 * 1024
        try:
            with Image.open(request.stream, formats=["JPEG"]) as uploaded:
                if max(uploaded.size) > 960:
                    return Response("画面尺寸过大。", status=400, mimetype="text/plain")
                image = uploaded.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            return Response("无法读取视频画面。", status=400, mimetype="text/plain")
        indoor = request.args.get("mode") == "indoor"
        if indoor and indoor_model is None:
            return Response("室内模型权重缺失。", status=503, mimetype="text/plain")
        if not inference_slot.acquire(blocking=False):
            return Response(status=429)
        try:
            segmented, hint = worker.submit(
                live_result, image, indoor, request.args.get("enhance") == "1",
            ).result()
        finally:
            inference_slot.release()
        result = BytesIO()
        segmented.save(result, format="JPEG", quality=85)
        server_ms = 1000 * (perf_counter() - started)
        if server_ms >= 300:
            app.logger.warning("Slow live frame: %.0f ms (%s, %sx%s)", server_ms,
                               "indoor" if indoor else "outdoor", image.width, image.height)
        return Response(result.getvalue(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-WalkScapes-Device": device.type,
                                 "X-WalkScapes-Server-Ms": f"{server_ms:.0f}",
                                 "X-WalkScapes-Direction": hint})

    @app.errorhandler(413)
    def too_large(_error):
        if request.endpoint == "live_frame":
            return Response("画面文件过大。", status=413, mimetype="text/plain")
        if request.endpoint == "video":
            return Response("请选择小于 100 MB 的视频。", status=413, mimetype="text/plain")
        return render_template_string(PAGE, error="请选择小于 25 MB 的照片。"), 413

    @app.post("/video")
    def video():
        upload = request.files.get("video")
        if not upload or not upload.filename:
            return Response("请选择一个视频。", status=400, mimetype="text/plain")
        extension = Path(upload.filename).suffix.lower()
        if extension not in {".mp4", ".mov", ".avi", ".webm"}:
            return Response("请选择 MP4、MOV、AVI 或 WebM 视频。", status=400, mimetype="text/plain")
        with TemporaryDirectory(prefix="walkscapes-video-") as directory:
            source = Path(directory) / ("source" + extension)
            destination = Path(directory) / "segmented.webm"
            upload.save(source)
            indoor = request.args.get("mode") == "indoor"
            if indoor and indoor_model is None:
                return Response("室内模型权重缺失。", status=503, mimetype="text/plain")

            def render_video_frame(image):
                with inference_slot:
                    return worker.submit(live_result, image, indoor, False).result()[0]

            try:
                segment_video(source, destination, infer, render_frame=render_video_frame)
                return Response(destination.read_bytes(), mimetype="video/webm")
            except ValueError as error:
                return Response(str(error), status=400, mimetype="text/plain")
            except (RuntimeError, cv2.error, OSError):
                app.logger.exception("Video segmentation failed")
                return Response("视频分割失败，请尝试另一个视频。", status=500, mimetype="text/plain")

    @app.route("/", methods=["GET", "POST"])
    def home():
        if request.method == "GET":
            return render_template_string(PAGE)
        request.max_content_length = 25 * 1024 * 1024
        photo = request.files.get("photo")
        if not photo or not photo.filename:
            return render_template_string(PAGE, error="请选择一张照片。"), 400
        try:
            with Image.open(photo.stream, formats=["JPEG", "PNG", "WEBP"]) as uploaded:
                image = uploaded.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            return render_template_string(PAGE, error="无法读取照片，请上传 JPG、PNG 或 WebP。"), 400

        mask = infer(image)
        legend = [
            (CLASSES_ZH[index], "#" + bytes(COLORS[index]).hex())
            for index in np.unique(mask)
        ]
        return render_template_string(PAGE, result={
            "original": image_url(image),
            "overlay": image_url(overlay(image, mask)),
            "mask": image_url(Image.fromarray(mask.astype(np.uint8))),
            "legend": legend,
        })

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the local WalkScapes webpage")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_app().run(host="127.0.0.1", port=args.port)
