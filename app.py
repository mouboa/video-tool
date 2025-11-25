import streamlit as st
import cv2
import numpy as np
import subprocess
import os
import tempfile
from PIL import Image

# -------------------------
# 【核心修改】定義靜態設定檔的路徑
# -------------------------
CONFIG_PATHS = {
    "LU": "configs/LU.txt",
    "LD": "configs/LD.txt",
    "RU": "configs/RU.txt",
    "RD": "configs/RD.txt",
}


# -------------------------
# 圖像疊加函式 (保持不變)
# -------------------------
def overlay_image(background_frame, overlay_img, x, y, w, h):
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
    overlay = overlay[0:y2-y1, 0:x2-x1]
    
    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    bg_slice = background_frame[y1:y2, x1:x2]
    
    for c in range(0, 3):
        bg_slice[:, :, c] = (alpha_s * overlay[:, :, c] + alpha_l * bg_slice[:, :, c])
    return background_frame

# -------------------------
# 解析設定檔 (參數從檔案物件改為檔案內容 bytes)
# -------------------------
def parse_config(config_content_bytes):
    frame_map = {}
    try:
        # 直接使用傳入的 bytes 內容
        lines = config_content_bytes.decode("utf-8-sig").splitlines()
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
# 核心處理邏輯 (保持不變)
# -------------------------
def process_video(video_path, frame_map, overlay_path=None):
    # ... (此處程式碼與前面版本相同，不重複貼出以節省篇幅) ...
    # 由於篇幅限制，請沿用您前一版本的 process_video 函式內容
    
    subscribe_img = None
    if overlay_path:
        subscribe_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if subscribe_img is None or subscribe_img.shape[2] != 4:
            st.warning("注意：疊圖檔案格式錯誤或是無 Alpha 通道，將略過疊圖步驟，僅執行去浮水印。")
            subscribe_img = None

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

            clean_frame = cv2.inpaint(processed_frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
            
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
# 網頁介面主程式 (主要修改區)
# -------------------------
def main():
    st.set_page_config(page_title="影片去水印工具", layout="centered")
    st.title("🎬 影片去浮水印工具 (預載配置版)")
    st.markdown("上傳影片，並選擇預載的座標設定檔 (LU/LD/RU/RD)。**無需再次上傳 TXT 檔。**")

    temp_paths = []
    
    with st.form("upload_form"):
        # 1. 影片和圖片
        video_file = st.file_uploader("1. 上傳影片 (MP4)", type=["mp4", "mov", "avi"])
        overlay_file = st.file_uploader("2. (選填) 上傳去背圖 (PNG)", type=["png"])

        st.subheader("3. 座標設定檔選取(浮水印起始位置L:左,U:上")
        
        # 移除檔案上傳欄位，改用選擇
        selected_key = st.selectbox(
            "請選擇要套用哪一個座標配置檔：",
            options=["--- 請選擇 ---", "LU", "LD", "RU", "RD"],
            index=0
        )
        st.caption("設定檔 (LU.txt, LD.txt等) 已經預先部署在伺服器上。")
        
        submitted = st.form_submit_button("開始處理")

    if submitted:
        try:
            # 檢查必填項目
            if not video_file:
                st.error("請上傳影片！")
                return
            if selected_key == "--- 請選擇 ---":
                st.error("請選擇一個座標配置檔 (LU/LD/RU/RD)！")
                return
            
            # --- 核心修改：讀取伺服器上的靜態檔案 ---
            config_server_path = CONFIG_PATHS.get(selected_key)
            
            if not os.path.exists(config_server_path):
                # 如果找不到檔案，通常是忘了提交到 GitHub
                st.error(f"❌ 錯誤：伺服器上找不到 [{selected_key}] 的設定檔 ({config_server_path})。請確認您已在 GitHub 提交了 /configs/{selected_key}.txt 檔案。")
                return
            
            # 從伺服器路徑讀取檔案內容
            with open(config_server_path, 'rb') as f:
                config_content_bytes = f.read()

            st.info(f"✅ 已選定：影片、疊圖 ({'已上傳' if overlay_file else '未上傳'})、預載座標檔 [{selected_key}]")

            # 儲存上傳的檔案到暫存區
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t_vid:
                t_vid.write(video_file.read())
                v_path = t_vid.name
                temp_paths.append(v_path)
            
            # 處理選填的圖片
            o_path = None
            if overlay_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t_img:
                    t_img.write(overlay_file.read())
                    o_path = t_img.name
                    temp_paths.append(o_path)

            # 解析設定檔 (傳入 bytes 內容)
            frame_map = parse_config(config_content_bytes)
            
            if frame_map:
                result_path = process_video(v_path, frame_map, o_path)
                
                if result_path:
                    st.success("🎉 處理完成！請下載影片。")
                    with open(result_path, "rb") as f:
                        st.download_button("下載影片", f, file_name=f"clean_video_{selected_key}.mp4")
                    temp_paths.append(result_path)
            
        except Exception as e:
            st.exception(e)
            st.error("處理過程中發生未知錯誤。")
        finally:
            # 清理所有暫存檔案
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)

if __name__ == "__main__":
    main()

