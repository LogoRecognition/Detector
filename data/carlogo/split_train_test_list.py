import os

def listdir(path, list_name):
    a = os.listdir(path+"/"+list_name)
    a.sort()
    b = []
    for file in a:
        b.append('CarLogo51/'+list_name+'/'+os.path.splitext(file)[0])
        #if os.path.splitext(file)[1] == '.txt':
           # b.append(path+'/'+list_name+'/'+file)
    return b
proportion = 1.0
def get_trainlist(path,brandlist):
    trainlist = []
    testlist = []
    for brand in brandlist:
        tmp = listdir(path, brand)
        print("len:"+str(len(tmp)))
        trainlist.extend(tmp[0:int(len(tmp)*proportion)]);
        print(len(trainlist))
        testlist.extend(tmp[int(len(tmp)*proportion):]);
        print(len(testlist))
    return trainlist,testlist

def write2file(fileName,lists):
    with open(fileName,'w+') as fp:
        for line in lists:
            fp.write(line+'\n')
        fp.close()



if __name__ == '__main__':
    pwd = os.getcwd()
    pwd += "/CarLogo51"
    brandlist = ['Acura',	'Alpha-Romeo',	'Aston-Martin',	'Audi',	'Bentley',	'Benz',	'BMW',	'Bugatti',	'Buick',	'nike',	'adidas',	'vans',	'converse',	'puma',	'nb',	'anta',	'lining',	'pessi',	'yili',	'uniquo',	'coca',	'Haier',	'Huawei',	'Apple',	'Lenovo',	'McDonalds',	'Amazon'];
    trainlist,testlist = get_trainlist(pwd, brandlist)
    #print(testlist)
    write2file(os.getcwd()+"/train_list_all.txt", trainlist)
    write2file(os.getcwd()+"/test_list_all.txt", testlist)
