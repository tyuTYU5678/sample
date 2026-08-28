import os
import time
from pathlib import Path
from ultralytics import YOLO

model = YOLO("/home/hiraga/work/groundingdino/train_by_yolov8n/set001/weights/best.pt")
        
def run_inference(dir_output, image_path):
    global model
    
    # 1. 画像ファイルの存在確認
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません -> {image_path}")
        return

    try:
        filename = os.path.basename(image_path)
    
        t0 = time.perf_counter()
    
         # 3. 推論の実行
        # save=True で検出枠を描画した画像が runs/detect/predict/ に自動保存されます
        results = model.predict(
            source=image_path,
            imgsz=640,  # 26n標準の入力サイズ
            conf=0.25,  # 検出のしきい値（信頼度25%以上を表示）
            save=True,  # 結果画像を保存する
            project=dir_output,  # 保存先の親フォルダ名
            name="",  # その下のサブフォルダ名 
            exist_ok=True           
        )

        t1 = time.perf_counter()

        # 4. 検出結果の保存先と詳細をコンソールに表示
        for result in results:
            print(f"【成功】結果画像の保存先: {result.save_dir}")

            # 検出されたオブジェクトの情報を取得
            boxes = result.boxes
            print(f"検出されたオブジェクト数: {len(boxes)}個")

            for box in boxes:
                cls_id = int(box.cls.item())  # クラスID
                cls_name = model.names[cls_id]  # クラス名（person, car など）
                conf = box.conf.item()  # 信頼度スコア（0.0 ~ 1.0）
                xyxy = box.xyxy.tolist()[0]  # 座標 [左上x, 左上y, 右下x, 右下y]

                print(
                    f" - [{cls_name}] (信頼度: {conf:.2f}) 座標: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                )

    except Exception as e:
        print(f"【失敗】エラーが発生しました: {e}")

    return 1000 * (t1 - t0) # msec   

if __name__ == "__main__":
    dir_input = "/home/hiraga/work/groundingdino/data/set001/test"
    dir_output = "/home/hiraga/work/groundingdino/data/set001/output"
    os.makedirs(dir_output, exist_ok=True)
    target_dir = Path(dir_input)
    file_list = [f.name for f in target_dir.glob("??????.jpg")]
    print(file_list)
    lst_msec = []
    for k, fn in enumerate(file_list):
        path = target_dir / fn
        print(path)
        dt = run_inference(dir_output, path)
        lst_msec.append(dt)
        
        
    path_elapse = Path(dir_output) / "elapse.txt"
    with open(path_elapse, "w", encoding="utf-8") as f:
        for msec in lst_msec:
            f.write("%8.3f\n" % msec)          
        
        
