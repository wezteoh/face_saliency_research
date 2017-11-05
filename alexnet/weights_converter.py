import _pickle as cPickle

snapshot = {}
snapshot["conv1W"] = net_data['conv1'][0]
snapshot["conv1b"] = net_data['conv1'][1]
snapshot["conv2W"] = net_data['conv2'][0]
snapshot["conv2b"] = net_data['conv2'][1]
snapshot["conv3W"] = net_data['conv3'][0]
snapshot["conv3b"] = net_data['conv3'][1]
snapshot["conv4W"] = net_data['conv4'][0]
snapshot["conv4b"] = net_data['conv4'][1]
snapshot["conv5W"] = net_data['conv5'][0]
snapshot["conv5b"] = net_data['conv5'][1]
snapshot["fc1W"] = net_data['fc6'][0]
snapshot["fc1b"] = net_data['fc6'][1]
snapshot["fc2W"] = net_data['fc7'][0]
snapshot["fc2b"] = net_data['fc7'][1]
snapshot["fc3W"] = net_data['fc8'][0]
snapshot["fc3b"] = net_data['fc8'][1]

cPickle.dump(snapshot,  open("alexnet_weights.pkl", "wb"))