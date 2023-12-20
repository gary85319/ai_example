模型可調整參數:
batch_size = 128(批次大小)#批次愈大，訓練愈快，電腦愈容易炸裂
num_layers=4(模型疊加層數)#疊加多不一定準，但依定會變慢
load_model=False(是否載入模型，欲訓練設False)
num_epochs = 1800(訓訓練次數，建議500以上)
如果要更換欲預測文本，請改這行line(12):with open("r.txt", "r", encoding="utf-8") as file:
    text = file.read()
    text = re.sub(r'[\n,，。\t()]+', '', text)
預設是r.txt，你可以換成任意純文字檔案
