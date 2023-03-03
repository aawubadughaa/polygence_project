def imgarr(word_label)
    img = cv2.imread(word_label)
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    imgtor = torch.from_numpy(res)