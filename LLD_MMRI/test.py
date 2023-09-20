import json
import os
import SimpleITK as sitk
import numpy as np
import torch
from net import net2, net1

path = r'../LLD-MMRI2023/test'  # 需要预测的图像

img_name_CA = 'C+A'
img_name_CD = 'C+Delay'
img_name_CV = 'C+V'
img_name_Cp = 'C-pre'
img_name_DWI = 'DWI'
img_name_IP = 'In Phase'
img_name_OP = 'Out Phase'
img_name_T2 = 'T2WI'


def get_array(path, file):
    mri = sitk.ReadImage(os.path.join(path, file))
    # spacing1 = list(mri.GetSpacing())
    mri_array = sitk.GetArrayFromImage(mri)
    mri_array = (mri_array - np.mean(mri_array)) / np.std(mri_array)
    new_mri = np.zeros((mri_array.shape[0] + 4, mri_array.shape[1], mri_array.shape[2]), dtype='float64')
    new_mri[2:new_mri.shape[0] - 2, :, :] = mri_array
    new_mri[0, :, :] = mri_array[0, :, :]
    new_mri[0, :, :] = mri_array[0, :, :]
    new_mri[new_mri.shape[0] - 1, :, :] = mri_array[-2, :, :]
    new_mri[new_mri.shape[0] - 2, :, :] = mri_array[-2, :, :]
    mri_size = new_mri.shape
    return new_mri, mri_size


def test(model):
    model.eval()
    test_path = os.listdir(os.path.join(path, img_name_CA))

    per = np.zeros([7])
    per_lab = [0, 0, 0, 0, 0, 0, 0]
    per_all = []
    per_middle_all = []
    per_lab_all = []
    ids = []
    file_index = 0

    for file in test_path:

        mri_CA, mri_size_CA = get_array(os.path.join(path, img_name_CA), file)
        mri_CD, mri_size_CD = get_array(os.path.join(path, img_name_CD), file)
        mri_CV, mri_size_CV = get_array(os.path.join(path, img_name_CV), file)
        mri_Cp, mri_size_Cp = get_array(os.path.join(path, img_name_Cp), file)
        mri_DWI, mri_size_DWI = get_array(os.path.join(path, img_name_DWI), file)
        mri_IP, mri_size_IP = get_array(os.path.join(path, img_name_IP), file)
        mri_OP, mri_size_OP = get_array(os.path.join(path, img_name_OP), file)
        mri_T2, mri_size_T2 = get_array(os.path.join(path, img_name_T2), file)

        per = 0
        per_lab = [0, 0, 0, 0, 0, 0, 0]
        with torch.no_grad():

            for i in range(1, mri_size_CA[0] - 4 - 1):
                mri_array = np.zeros((5 * 8, mri_size_CA[1], mri_size_CA[1]), dtype='float64')

                mri_array[0:5, :, :] = mri_CA[i:i + 5, :, :]
                mri_array[5:10, :, :] = mri_CD[i:i + 5, :, :]
                mri_array[10:15, :, :] = mri_CV[i:i + 5, :, :]
                mri_array[15:20, :, :] = mri_Cp[i:i + 5, :, :]
                mri_array[20:25, :, :] = mri_DWI[i:i + 5, :, :]
                mri_array[25:30, :, :] = mri_IP[i:i + 5, :, :]
                mri_array[30:35, :, :] = mri_OP[i:i + 5, :, :]
                mri_array[35:40, :, :] = mri_T2[i:i + 5, :, :]

                ct_tensor = torch.FloatTensor(mri_array).cuda()

                ct_tensor = ct_tensor.unsqueeze(dim=0)
                # print(ct_tensor.shape)
                output = model(ct_tensor)
                # print(output.shape)
                output = torch.softmax(output, dim=1)
                per_lab[np.argmax(output.cpu().numpy())] += 1
                if i == mri_size_CA[0] // 2 - 1:
                    per_middle = output.cpu().numpy()

                per = per + output

        per = per / (mri_size_CA[0] - 6)
        id = (file.split('.')[0]).split('_')[0]
        per_lab_all.append(np.array(per_lab))
        per_middle_all.append(np.array(per_middle))
        per_all.append(per.cpu().numpy())
        ids.append(id)

    return per_all, per_lab_all, per_middle_all, ids


def write_json(per1, per2, per_lab1, per_lab2, per_middle1, per_middle2, ids):
    score_list = []
    per1 = np.array(per1)
    per2 = np.array(per2)
    per_lab2 = np.array(per_lab2)
    per_lab1 = np.array(per_lab1)
    per_lab_all = (np.array(per_lab1) + np.array(per_lab2))

    for index in range(len(ids)):

        id = ids[index]
        per = (per1[index] + per2[index]) / 2
        if np.argmax(per1[index]) != np.argmax(per2[index]):
            if (per_lab_all[index] == per_lab_all[index].max()).sum() == 1:
                per = per_lab_all[index] / np.sum(per_lab_all[index])
            else:
                per = (per_middle1[index] + per_middle2[index]) / 2

        score = list(per.tolist())
        pred = int(np.argmax(per))
        pred_info = {
            'image_id': id,
            'prediction': pred,
            'score': score,
        }
        print(pred, end=', ')
        score_list.append(pred_info)

    json_data = json.dumps(score_list, indent=4)
    save_name = os.path.join('./MediSegLearner.json')
    file = open(save_name, 'w')
    file.write(json_data)
    file.close()


if __name__ == '__main__':

    net = net1.U_Net().cuda()

    net.load_state_dict(torch.load('./model/net1/epoch_499_acc_0.75.pth'))
    per2, per_lab2, per_middle1, id = test(net)

    net = net2.U_Net().cuda()
    net.load_state_dict(torch.load('./model/net2/epoch_495_acc_0.7670940170940171.pth'))
    per5, per_lab5, per_middle2, id = test(net)

    write_json(per2, per5, per_lab2, per_lab5, per_middle1, per_middle2, id)

