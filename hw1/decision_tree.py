#!/usr/local/bin/python
# INF 552 Assignment 1
# Name: Zongdi Xu
# USC ID: 5900-5757-70

import re
from math import log


def main():
    dataset, tree, attribute = read_data('dt-data.txt')
    test = {'Occupied': 'Moderate', 'Price': 'Cheap', 'Music': 'Loud',
            'Location': 'City-Center', 'VIP': 'No', 'Favorite Beer': 'No'}

    recursive_build(dataset, tree, attribute)
    recursive_print(tree)
    print 'The prediction test result of', test, 'is:', recursive_predict(
        test, dataset, tree)


def read_data(filename):
    input_f = open(filename, 'r')
    data = input_f.readlines()

    predictor = [re.sub(r'^\s+|\s+$', '', word)
                 for word in list(filter(None, re.split(r'\(|\)|,|\n', data[0])))]

    attribute = {'num': len(predictor), 'caption': predictor, 'isChosen': [
        False]*len(predictor)}

    tree = {'__list__': range(len(data)-2)}

    dataset = [list(filter(None, re.split(' |:|,|;|\n', rec)))[1:]
               for rec in data[2:]]

    return dataset, tree, attribute


def entropy(dataset, li, observe, target):
    attr_cnt = {}
    sum = 0
    for id in li:
        if not attr_cnt.has_key(dataset[id][observe]):
            attr_cnt[dataset[id][observe]] = [id]
        else:
            attr_cnt[dataset[id][observe]].append(id)
        sum += 1

    e = 0.0

    for value in attr_cnt.itervalues():
        probability = 1.0*len(value)/sum
        if observe == target:
            e += probability*log(1.0/probability)/log(2.0)
        else:
            e += probability*entropy(dataset, value, target, target)
    return e


def recursive_build(dataset, node, attribute):

    target = attribute['num']-1
    target_entropy = entropy(
        dataset, node['__list__'], target, target)

    chosen = -1
    max_inf_gain = 0.0
    for observe in range(attribute['num']):
        if not attribute['isChosen'][observe]:
            branch_entropy = entropy(
                dataset, node['__list__'], observe, target)
            inf_gain = target_entropy - branch_entropy

            if inf_gain > max_inf_gain:
                max_inf_gain = inf_gain
                chosen = observe

    # print target_entropy, chosen

    if chosen > -1:
        attribute['isChosen'][chosen] = True
        node['__label__'] = attribute['caption'][chosen]

        for id in node['__list__']:
            if not node.has_key(dataset[id][chosen]):
                node[dataset[id][chosen]] = {'__list__': [id]}
            elif not node[dataset[id][chosen]].has_key('__list__'):
                node[dataset[id][chosen]]['__list__'] = [id]
            else:
                node[dataset[id][chosen]]['__list__'].append(id)

        for (key, value) in node.iteritems():
            if not key == '__list__' and not key == '__label__':
                recursive_build(dataset, value, attribute)

        attribute['isChosen'][chosen] = False
    else:
        node['__res__'] = dataset[node['__list__'][0]][target]
        # node[dataset[node['__list__'][0]][target]] = {'__list__':list(node['__list__'])}


def recursive_print(node, level=0):
    if node.has_key('__label__'):
        print node['__label__']

    for key in node.keys():
        if not key == '__list__' and not key == '__label__':
            if node.has_key('__res__'):
                print node['__res__']
            else:
                print '  '*(level+1)+key+':',
                recursive_print(node[key], level+1)


def recursive_predict(testdata, dataset, node):
    if node.has_key('__res__'):
        return node['__res__']

    try:
        next = testdata.get(node['__label__'])
        return recursive_predict(testdata, dataset, node[next])
    except:
        return '[ERROR]'


if __name__ == '__main__':
    main()
