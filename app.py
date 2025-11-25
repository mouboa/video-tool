import streamlit as st
import cv2
import numpy as np
import subprocess
import os
import tempfile
from PIL import Image

# -------------------------
# 圖像疊加函式
# -------------------------
def overlay_image(background_frame, overlay_img, x, y, w, h):
    # 如果沒有傳入疊圖，直接回傳原背景
    if overlay_img is None:
        return background_frame

    if w <= 0 or h <= 0:
        return background_frame
    try:
        img_pil = Image.fromarray(cv2.cvtColor(overlay_img, cv2.COLOR_BGRA2RGBA))
        img_pil = img_pil.resize((w, h), Image.Resampling.LANCZOS)
        overlay = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        return background_frame

    y1, y2 = max(0, y), min(background_frame.shape[0], y + h)
    x1, x2 = max(0, x), min(background_frame.shape[1], x + w)

    # 確保疊加圖片尺寸與目標區域匹配
    overlay = overlay[0:y2-y1, 0:x2-x1]

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    bg_slice = background_frame[y1:y2, x1:x2]

    for c in range(0, 3):
        bg_slice[:, :, c] = (alpha_s * overlay[:, :, c] + alpha_l * bg_slice[:, :, c])
    return background_frame

# -------------------------
# 解析設定檔
# -------------------------
def parse_config(config_content):
    frame_map = {}
    try:
        lines = config_content.decode("utf-8-sig").splitlines()
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                start, end, x, y, w, h = map(int, parts[:6])
                for i in range(start, end + 1):
                    if i not in frame_map:
                        frame_map[i] = []
                    frame_map[i].append((x, y, w, h))
        return frame_map
    except Exception as e:
        st.error(f"設定檔解析失敗: {e}")
        return None

# -------------------------
# 核心處理邏輯 (修改：overlay_path 可為 None)
# -------------------------
def process_video(video_path, frame_map, overlay_path=None):
    # 1. 嘗試讀取疊圖 (如果有傳入路徑)
    subscribe_img = None
    if overlay_path:
        subscribe_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if subscribe_img is None or subscribe_img.shape[2] != 4:
            st.warning("注意：疊圖檔案格式錯誤或是無 Alpha 通道，將略過疊圖步驟，僅執行去浮水印。")
            subscribe_img = None # 強制設為 None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tfile_out.name
    tfile_out.close()

    temp_silent = tmp_video_path + "_silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_silent, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_idx = 0
    inpaint_radius = 3
    padding = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_map:
            bboxes = frame_map[frame_idx]
            mask = np.zeros((height, width), dtype=np.uint8)
            processed_frame = frame.copy()

            for (x, y, w, h) in bboxes:
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                mask[y1:y2, x1:x2] = 255

            # 1. 先去浮水印 (Inpaint)
            clean_frame = cv2.inpaint(processed_frame, mask, inpaint_radius, cv2.INPAINT_TELEA)

            # 2. 如果有圖片，才執行疊加
            if subscribe_img is not None:
                for (x, y, w, h) in bboxes:
                    clean_frame = overlay_image(clean_frame, subscribe_img, x, y, w, h)

            out.write(clean_frame)
        else:
            out.write(frame)

        frame_idx += 1
        if frame_idx % 10 == 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            status_text.text(f"處理進度: {int(frame_idx/total_frames*100)}%")

    cap.release()
    out.release()

    status_text.text("影像處理完成，正在合併音訊...")
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_silent,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            tmp_video_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_silent)
        return tmp_video_path
    except Exception as e:
        st.error(f"FFmpeg 合併失敗: {e}")
        return temp_silent

# -------------------------
# 網頁介面主程式
# -------------------------
def main():
    st.set_page_config(page_title="影片去水印工具", layout="centered")
    st.title("🎬 影片去浮水印工具 (純淨版)")
    st.markdown("上傳影片與座標設定檔 (TXT)。**圖片為選填**，若不傳圖片則單純去除浮水印。")

    with st.form("upload_form"):
        video_file = st.file_uploader("1. 上傳影片 (MP4)", type=["mp4", "mov", "avi"])
        config_file = st.file_uploader("2. 上傳座標設定檔 (TXT)", type=["txt"])
        overlay_file = st.file_uploader("3. (選填) 上傳疊圖", type=["png"]) # 標記為選填

        submitted = st.form_submit_button("開始處理")

    if submitted:
        # 修改判斷條件：只要有 影片 和 設定檔 即可
        if video_file and config_file:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t_vid:
                t_vid.write(video_file.read())
                v_path = t_vid.name

            # 處理選填的圖片
            o_path = None
            if overlay_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t_img:
                    t_img.write(overlay_file.read())
                    o_path = t_img.name

            frame_map = parse_config(config_file.getvalue())

            if frame_map:
                # 傳入路徑 (o_path 可能為 None)
                result_path = process_video(v_path, frame_map, o_path)

                if result_path:
                    st.success("處理完成！請下載影片。")
                    with open(result_path, "rb") as f:
                        st.download_button("下載影片", f, file_name="clean_video.mp4")
                    os.remove(result_path)

            os.remove(v_path)
            if o_path:
                os.remove(o_path)
        else:
            st.error("請至少上傳「影片」和「座標設定檔」！")

if __name__ == "__main__":
    main()