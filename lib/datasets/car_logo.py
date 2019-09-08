import datasets
import os
import datasets.imdb
import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class car_logo(datasets.imdb):
    def __init__(self, image_set, data_path=None):
        datasets.imdb.__init__(self, 'car_logo_' + image_set)
        self._image_set = image_set
        self._data_path = data_path
        self._classes = ('__background__', # always index 0
                         'Acura',	'Alpha-Romeo',	'Aston-Martin',	'Audi',	'Bentley',	'Benz',	'BMW',	'Bugatti',	'Buick',	'nike',	'adidas',	'vans',	'converse',	'puma',	'nb',	'anta',	'lining',	'pessi',	'yili',	'uniquo',	'coca',	'Haier',	'Huawei',	'Apple',	'Lenovo',	'McDonalds',	'Amazon')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._data_path), \
                'Image Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index+self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.

        """
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt') 
    
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        #gt_roidb = self._load_annotation()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_annotation(self,index):
        """
        Load image and bounding boxes info from txt format.
        """

        filename = os.path.join(self._data_path, 'annotations', index[10:] + '.jpg.txt')
        print(filename)
        f = open(filename)
        lines = f.readlines()

        num_objs = len(lines)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, line in enumerate(lines):
            sep_line = line.strip().split(' ')

            x1 = float(sep_line[0])
            y1 = float(sep_line[1])

            x2 = float(sep_line[0])+float(sep_line[2])
            y2 = float(sep_line[1])+float(sep_line[3])
            cls = int(sep_line[4])
            print(cls)
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
    
    

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} logo results file'.format(cls)
            filename = self._data_path + '/result/_det_' + self._image_set + '_{:s}.txt'.format(cls)
            
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
    
    def parse_rec(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        objects = []
        for ix,line in enumerate(lines):
            sep_line = line.strip().split(' ')
            obj_struct = {}
            print(filename,sep_line[4])
            obj_struct['name'] = self._classes[int(sep_line[4])]
            
            x1 = float(sep_line[0])
            y1 = float(sep_line[1])

            x2 = float(sep_line[0])+float(sep_line[2])
            y2 = float(sep_line[1])+float(sep_line[3])
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(x1),
                              int(y1),
                              int(x2),
                              int(y2)]
            objects.append(obj_struct)
        return objects
    def voc_eval(self,detpath,imagesetfile,annopath,classname,cachedir,ovthresh=0.5,use_07_metric=False):

        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
       
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()

        print(detpath)
        #print(lines)
        imagenames = [x.strip() for x in lines]
        recs = {}
        for i, imagename in enumerate(imagenames):
            iname = imagename[10:]
            recs[imagename] = self.parse_rec(annopath.format(iname))
          
        # extract gt objects for this class
    	class_recs = {}
        #print(classname)
    	npos = 0
    	for imagename in imagenames:
            R = []
            for obj in recs[imagename]:
                #print(obj['name'],classname)
                if obj['name'] == classname:
                    R.append(obj)
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
           
        # read dets
        detfile = detpath.format(classname)
        #print(detfile)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
            
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            #print(sorted_scores)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]
            #print(image_ids)
            
            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                #print(image_ids[d])
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                #print(BBGT)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                    #print(inters)
                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.
            # compute precision recall
            fp = np.cumsum(fp)
            #print(fp)
            tp = np.cumsum(tp)
            #print(tp)
            #print(npos)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec)

        else:
            rec = -1
            prec = -1
            ap = -1

        #print(rec, prec, ap)
        return rec, prec, ap
    def voc_ap(self,rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    	return ap

    def _do_python_eval(self, output_dir = 'output'):
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            #filename = self._get_voc_results_file_template().format(cls)
            filename = filename = self._data_path + '/result/_det_' + self._image_set + '_{:s}.txt'.format(cls)
            imagesetfile = self._data_path+'/'+self._image_set+'.txt'
            annopath = self._data_path+'/annotations/{:s}.jpg.txt'
            #print(imagesetfile)

            rec, prec, ap = self.voc_eval(filename, imagesetfile, annopath, cls, self._data_path+'/'+'cachedir', 0.5, False)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

           

        
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        #print(output_dir)
        self._do_python_eval(output_dir)
        #print(self._data_path)
        #for cls_ind, cls in enumerate(self.classes):

            #print(cls_ind, cls)
            #for im_ind, index in enumerate(self.image_index):
                #print(im_ind, index)
       	        #dets = all_boxes[cls_ind][im_ind]
                #print(dets)
        #self._write_voc_results_file(all_boxes)
        #self._do_python_eval(output_dir)
        #if self.config['matlab_eval']:
        #    self._do_matlab_eval(output_dir)
        #if self.config['cleanup']:
        #    for cls in self._classes:
        #        if cls == '__background__':
        #            continue
        #        filename = self._get_voc_results_file_template().format(cls)
        #        os.remove(filename)

    
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    import datasets.car_logo
    d = datasets.car_logo('test_list_27', './data/carlogo')
    res = d.roidb
    from IPython import embed; embed()
