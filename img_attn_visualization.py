import torch
import pickle, json
import numpy as np
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from itertools import chain
import cv2
from progressbar import ProgressBar

pbar = ProgressBar()

#image attention data
att_data = []
with open('/home/user0x00/Project/temp/image_attn_val.pkl', 'rb') as fr:
    try:
        while True:
            att_data.append(pickle.load(fr))
    except EOFError:
        pass

#Question and answer data from mcan-small model
with open('/home/user0x00/Project/temp/result_run_epoch13.pkl_31907550.json','r') as f:
    pred_data = json.load(f)

predByQid = {}
for entry in pred_data:
    predByQid[entry['question_id']] = entry['answer']


with open('/home/user0x00/project/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json') as fd:
    ques_data = json.load(fd)

ques_data = ques_data['questions']
questionImageById = {}
for entry in ques_data:
    questionImageById[entry['question_id']] = (entry['image_id'], entry['question'])

qix = {}
for i in range (len(ques_data)):
    qix[ques_data[i]['question_id']] = i

#subset of questions chosen for evaluation
with open('/home/user0x00/Project/questionsForUI.pkl', 'rb') as f:
    qids = pickle.load(f)
qids = list(chain.from_iterable(qids))

for i in range pbar((len(qids))):
    ques_id = qids[i] #use this to retrieve img from dataset.
    quesIndex = qix[ques_id]
    attIndex_i = quesIndex//32
    attIndex_j = quesIndex%32
    att = att_data[attIndex_i][attIndex_j]
    image_prefix = "COCO_val2014_000000000000"  # general format of validation set images
    image_path = "/home/user0x00/project/val2014/"
    image_features_path = '/home/user0x00/Project/val2014/'
    image_id = questionImageById[ques_id][0]  # replace last n characters of image_prefix with this
    image_prefix = image_prefix[:-(len(str(image_id)))]
    final_img_name = image_path + image_prefix + str(image_id) + ".jpg"
    final_img_feat_name = image_features_path + image_prefix +str(image_id) + ".jpg.npz"
    question = questionImageById[ques_id][1]
    img_feat = np.load(final_img_feat_name)
    bboxes = img_feat['bbox']

    plt.rcParams['figure.figsize'] = (8.0, 8.0)
    f, ax = plt.subplots()
    plt.suptitle("mcan-vqa::{} : {} ".format(question, predByQid[ques_id]))
    im = cv2.imread(final_img_name)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')
    gca = ax
    gca.axis('off')
    att = att.flatten()
    shape = (img.shape[0], img.shape[1], 1)
    A = np.zeros(shape)
    for k in range(len(bboxes)):
        bbox = bboxes[k].astype(int)
        A[bbox[1]:bbox[3],bbox[0]:bbox[2]] += att[k]
    A /= np.max(A)
    A = A * img + (1.0-A)*255
    A = A.astype('uint8')
    bbox = bboxes[np.argmax(att)]
    gca.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=1))
    gca.imshow(A, interpolation = 'bicubic')
    gca.axis('off')
    plt.tight_layout()
    plt.savefig("{}.jpg".format(i))

