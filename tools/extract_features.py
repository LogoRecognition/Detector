import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
CLASSES = ('__background__', # always index 0
                         'Acura',       'Alpha-Romeo',  'Aston-Martin', 'Audi', 'Bentley',      'Benz', 'BMW',  'Bugatti',      'Buick',        'nike', 'adidas',       'vans', 'converse',     'puma', 'nb',   'anta', 'lining',       'pessi',        'yili', 'uniquo',       'coca', 'Haier',        'Huawei',       'Apple',        'Lenovo',       'McDonalds',    'Amazon')
def vis_detections(im, class_name, dets,ax, image_name,fc7, brands,thresh=0.0):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    print(len(inds))
    for i in inds:
	print(i)
	param = fc7[i] 
	param = np.array(param)
        np.save('/home/CarLogo/features/'+image_name[:-4]+'.npy', param)
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        brands.append(class_name)
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.savefig('/home/CarLogo/detect/'+class_name+'_'+image_name)
    plt.axis('off')

    plt.tight_layout()
    plt.draw()

class extractor():
    def __init__(self):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
		# init session
        gpu_options = tf.GPUOptions(allow_growth=True)
    	self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    	# load network
    	self.net = get_network('VGGnet_test')
    	# load model
    	self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
#    	model = '/home/CarLogo/Faster_RCNN_TF/output/default/car_logo_train_list_27/VGGnet_fast_rcnn_iter_70000_test.ckpt'
	model = '/home/CarLogo/Faster_RCNN_TF/output/default/car_logo_train_list_all/VGGnet_fast_rcnn_iter_70000.ckpt' 
   	self.saver.restore(self.sess, model)

    	#sess.run(tf.initialize_all_variables())

    	print '\n\nLoaded network {:s}'.format(model)

    	im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    	for i in xrange(2):
            _, _, _= im_detect(self.sess, self.net, im)
    def get_feature(self,image_name):
    	#im_file = os.path.join(cfg.DATA_DIR, sys.argv[1], image_name)
    	im_file = os.path.join(cfg.DATA_DIR,'demo', image_name)
	im = cv2.imread(im_file)
    	timer = Timer()
    	timer.tic()
    	scores, boxes,fc7 = im_detect(self.sess, self.net, im)
    	print(fc7.shape)
	timer.toc()
    	print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    	im = im[:, :, (2, 1, 0)]
    	fig, ax = plt.subplots(figsize=(12, 12))
    	ax.imshow(im, aspect='equal')
    	CONF_THRESH = 0.8
    	NMS_THRESH = 0.3
        brands=[]
    	for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
           # brands.append(cls)
            vis_detections(im, cls, dets, ax, image_name, fc7,brands, thresh=CONF_THRESH)
        return brands
if __name__ == '__main__':
    e = extractor()
    filename= os.path.join(cfg.DATA_DIR,sys.argv[1])
    
    print('loading files from {}'.format(filename))
    im_names=[]
    for root, dirs, files in os.walk(filename):
        print(files)
        im_names = files
    im_names = ['0024.jpg', '0075.jpg', '0084.jpg',
                '0093.jpg']
    for image_name in im_names:
    	brands = e.get_feature(image_name)
        print(brands)


