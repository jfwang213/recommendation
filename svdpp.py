import numpy as np
import random


class SVD:
    def __init__(self, _k, iter_num=10, l=0.01):
        self.class_num = _k
        self.iter_num = iter_num
        self.l = l
        
    def fit(self, data):
        user_dict = {}
        item_dict = {}
        self.user_rate_items = {}
        
        sum_rate = 0
        for rate in data:
            #uid, itemid, score
            if rate[0] not in user_dict:
                user_dict[rate[0]] = [0, 0]
            if rate[1] not in item_dict:
                item_dict[rate[1]] = [0, 0]
            t = user_dict[rate[0]]
            t[0] += 1
            t[1] += rate[2]
            t = item_dict[rate[1]]
            t[0] += 1
            t[1] += rate[2]
            sum_rate += rate[2]
            if rate[0] not in self.user_rate_items:
                self.user_rate_items[rate[0]] = set()
            self.user_rate_items[rate[0]].add(rate[1])
        self.aver_rate = float(sum_rate) / len(data)
        print "aver rate", self.aver_rate
            
        self.user_diff_aver = {}
        self.item_diff_aver = {}
        self.vec_users = {}
        self.vec_items = {}
        self.item_y = {}
        for k, v in user_dict.iteritems():
            self.user_diff_aver[k] = float(v[1])/v[0] - self.aver_rate
            self.vec_users[k] = np.asarray([(1.0/30)**0.5] * self.class_num)
            
        for k, v in item_dict.iteritems():
            self.item_diff_aver[k] = float(v[1])/v[0] - self.aver_rate
            self.item_y[k] = np.zeros(self.class_num)
            self.vec_items[k] = np.asarray([(1.0/30)**0.5] * self.class_num)
        
        print "iter all", self.iter_num
        index = range(len(data))
        for i in range(self.iter_num):
            print "iter", i
            random.shuffle(index)
            gamma = 0.1 / (i + 1)
            for j in index:
                uid, itemid, score = data[j]
                user_rate = self.user_rate_items[uid]
                if len(user_rate) > 1:
                    other_item_ratio = (len(user_rate) - 1) ** -0.5
                else:
                    other_item_ratio = 0
                vec_user = self.vec_users[uid]
                vec_item = self.vec_items[itemid]
                user_diff = self.user_diff_aver[uid]
                item_diff = self.item_diff_aver[itemid]
                sumy = np.zeros(self.class_num)
                for t_iid in user_rate:
                    if t_iid != itemid:
                        sumy += self.item_y[t_iid]
                sumy *= other_item_ratio
                rscore = self.aver_rate + user_diff + item_diff + vec_item.T.dot(vec_user + sumy)
                escore = score - rscore
                old_vec_item = vec_item
                
                self.vec_items[itemid] = vec_item + gamma * (2 * escore * (vec_user + sumy) - 2 * self.l * vec_item)
                self.vec_users[uid] = vec_user + gamma * (2 * escore * old_vec_item - 2 * self.l * vec_user)
                
                self.user_diff_aver[uid] = user_diff + gamma * (2 * escore - 2 * self.l * user_diff)
                self.item_diff_aver[itemid] = item_diff + gamma * (2 * escore - 2 * self.l * item_diff)
                
                for t_iid in user_rate:
                    if t_iid != itemid:
                        item_y = self.item_y[t_iid]
                        self.item_y[t_iid] = item_y + gamma * (2 * escore * other_item_ratio * vec_item - 2 * self.l * item_y * other_item_ratio)
    def evaluate(self, data):
        sum_diff = 0.0
        num = 0
        for uid, itemid, score in data:
            if uid in self.user_diff_aver and itemid in self.item_diff_aver:
                user_rate = self.user_rate_items[uid]

                other_item_ratio = len(user_rate) ** -0.5
                
                sumy = np.zeros(self.class_num)
                for t_iid in user_rate:
                    if t_iid != itemid:
                        sumy += self.item_y[t_iid]
                sumy *= other_item_ratio
                rscore = self.aver_rate + self.user_diff_aver[uid] + self.item_diff_aver[itemid] + self.vec_items[itemid].T.dot(self.vec_users[uid] + sumy)
                sum_diff += (score - rscore) ** 2
                num += 1
            
        return (sum_diff/num) ** 0.5, num
    
if __name__ == '__main__':
    import sys
    datafile = file(sys.argv[1])
    testfile = file(sys.argv[2])
    data = []
    for line in datafile:
        seps = line.split()
        data.append([int(seps[0]), int(seps[1]), int(seps[2])])
    datafile.close()
    
    clf = SVD(30, iter_num=30)
    clf.fit(data)
    print "fit done"
    data = []
    for line in testfile:
        seps = line.split()
        data.append([int(seps[0]), int(seps[1]), int(seps[2])])
    testfile.close()
    print clf.evaluate(data)
    
                
