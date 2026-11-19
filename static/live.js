const live = document.querySelector('#live');
const source = live.querySelector('#live-source');
const output = live.querySelector('#live-overlay');
const outputContext = output.getContext('2d');
const cameraButton = live.querySelector('#start-camera');
const fileInput = live.querySelector('#live-file');
const stopButton = live.querySelector('#stop-live');
const liveStatus = live.querySelector('#live-status');
const mode = live.querySelector('#live-mode');
const enhance = live.querySelector('#live-enhance');
const enhanceControl = live.querySelector('#live-enhance-control');
const views = live.querySelector('#live-views');
const capture = document.createElement('canvas');
const captureContext = capture.getContext('2d', {alpha: false});
const frameDeadlineMs = 800;
let active;

function stopLive(message = '') {
  const previous = active;
  active = null;
  previous?.controller.abort();
  previous?.cancelNext?.();
  clearTimeout(previous?.expires);
  previous?.stream?.getTracks().forEach(track => track.stop());
  source.pause();
  source.srcObject = null;
  source.removeAttribute('src');
  source.load();
  if (previous?.url) URL.revokeObjectURL(previous.url);
  cameraButton.disabled = false;
  stopButton.disabled = true;
  views.hidden = true;
  outputContext.clearRect(0, 0, output.width, output.height);
  liveStatus.textContent = message;
}

function scheduleFrame(run) {
  if (active !== run || document.hidden || source.ended || run.busy || run.cancelNext) return;
  if (source.requestVideoFrameCallback) {
    const handle = source.requestVideoFrameCallback(() => analyzeFrame(run));
    run.cancelNext = () => source.cancelVideoFrameCallback(handle);
    return;
  }
  const handle = requestAnimationFrame(() => {
    run.cancelNext = null;
    if (active !== run) return;
    if (source.readyState >= 2 && source.currentTime !== run.mediaTime) {
      run.mediaTime = source.currentTime;
      analyzeFrame(run);
    } else {
      scheduleFrame(run);
    }
  });
  run.cancelNext = () => cancelAnimationFrame(handle);
}

async function analyzeFrame(run) {
  run.cancelNext = null;
  if (active !== run || document.hidden) return;
  run.busy = true;
  const started = performance.now();
  const deadline = AbortSignal.timeout(frameDeadlineMs);
  const scale = Math.min(1, 960 / Math.max(source.videoWidth, source.videoHeight));
  const width = Math.max(1, Math.round(source.videoWidth * scale));
  const height = Math.max(1, Math.round(source.videoHeight * scale));
  if (capture.width !== width || capture.height !== height) {
    capture.width = width;
    capture.height = height;
  }
  captureContext.drawImage(source, 0, 0, width, height);
  try {
    const frame = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', 0.8));
    if (!frame) throw new Error('无法采集视频帧。');
    const endpoint = new URL(live.dataset.frameUrl, location.href);
    endpoint.searchParams.set('mode', mode.value);
    if (mode.value === 'indoor' && enhance.checked) endpoint.searchParams.set('enhance', '1');
    const response = await fetch(endpoint, {
      method: 'POST', body: frame, signal: AbortSignal.any([run.controller.signal, deadline]),
    });
    if (response.status === 429) return;
    if (!response.ok) throw new Error(await response.text());
    const bitmap = await createImageBitmap(await response.blob());
    const now = performance.now();
    if (active === run && !document.hidden && now - started <= frameDeadlineMs) {
      if (output.width !== bitmap.width || output.height !== bitmap.height) {
        output.width = bitmap.width;
        output.height = bitmap.height;
      }
      outputContext.drawImage(bitmap, 0, 0);
      clearTimeout(run.expires);
      run.expires = setTimeout(() => {
        if (active === run && !source.paused && !source.ended) {
          outputContext.clearRect(0, 0, output.width, output.height);
          liveStatus.textContent = '等待最新分割画面…';
        }
      }, frameDeadlineMs);
      run.frames += 1;
      const elapsed = now - run.lastUpdate;
      if (elapsed >= 1000) {
        const fps = (1000 * run.frames / elapsed).toFixed(1);
        const device = response.headers.get('X-WalkScapes-Device') === 'cuda' ? 'GPU' : 'CPU';
        if (!source.ended) {
          const server = response.headers.get('X-WalkScapes-Server-Ms');
          liveStatus.textContent = `${device} · 分割 ${fps} FPS · 单帧往返 ${Math.round(now - started)} ms · 服务端 ${server} ms`;
        }
        run.frames = 0;
        run.lastUpdate = now;
      }
    }
    bitmap.close();
  } catch (error) {
    if (active === run && !deadline.aborted) stopLive(error.message || '视频分析失败。');
  } finally {
    run.busy = false;
    scheduleFrame(run);
  }
}

async function startLive(file) {
  stopLive();
  const run = {controller: new AbortController(), frames: 0, lastUpdate: performance.now()};
  active = run;
  cameraButton.disabled = !file;
  stopButton.disabled = false;
  liveStatus.textContent = file ? '正在打开视频…' : '正在打开摄像头…';
  try {
    if (file) {
      run.url = URL.createObjectURL(file);
      source.src = run.url;
    } else {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('请在本机浏览器中打开此网页并允许摄像头访问。');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false, video: {width: {ideal: 1280}, height: {ideal: 720}, frameRate: {ideal: 30}},
      });
      if (active !== run) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      run.stream = stream;
      source.srcObject = stream;
    }
    source.controls = Boolean(file);
    await source.play();
    if (active !== run) return;
    views.hidden = false;
    liveStatus.textContent = '正在实时分析…';
    scheduleFrame(run);
  } catch (error) {
    if (active !== run) return;
    const message = file ? '无法播放此视频，请试用浏览器支持的 MP4 或 WebM。'
      : `无法开启摄像头。请检查摄像头权限和连接。${error.name === 'Error' ? ' ' + error.message : ''}`;
    stopLive(message);
  }
}

cameraButton.addEventListener('click', () => startLive());
mode.addEventListener('change', () => {
  enhanceControl.hidden = mode.value !== 'indoor';
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) startLive(fileInput.files[0]);
  fileInput.value = '';
});
stopButton.addEventListener('click', () => stopLive('已停止'));
source.addEventListener('ended', () => {
  if (active) liveStatus.textContent = '播放结束';
});
source.addEventListener('play', () => {
  if (active && !views.hidden) scheduleFrame(active);
});
document.addEventListener('visibilitychange', () => {
  if (!active) return;
  if (document.hidden) {
    active.cancelNext?.();
    active.cancelNext = null;
    outputContext.clearRect(0, 0, output.width, output.height);
  } else {
    active.frames = 0;
    active.lastUpdate = performance.now();
    scheduleFrame(active);
  }
});
window.addEventListener('pagehide', () => stopLive());
