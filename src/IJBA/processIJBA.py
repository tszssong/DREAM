import sys, os
wkdir = '/home/ubuntu/zms/wkspace/DREAM/data/IJBA/'
for dirpath, dirnames, filenames in os.walk(wkdir):
    for filename in filenames:
        if '.txt' in filename:  
            print dirpath, filename
            txtname = os.path.join(dirpath, filename)
            f = open(txtname, 'r+')
            all_line = f.readlines()
            f.seek(0)
            f.truncate()
            for line in all_line:
                str1 = '/mnt/SSD/rongyu/data/'
                str2 = '/home/ubuntu/zms/wkspace/DREAM/data/'
                f.write(line.replace(str1, str2))
            f.close()
