if __name__ == '__main__':
    
    from ultralytics import YOLO
    
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    set_data= 'data_dior.yaml' #数据路径

    model = YOLO('best.pt')
    model.predict(data=set_data, batch=32, epochs=100, imgsz=640,workers=1,show=True,save=True,device=[1])     