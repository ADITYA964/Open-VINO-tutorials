
ie = IECore()

net = ie.read_network(model=model_xml, weights=model_bin)

input_blob = next(iter(net.input_info))

output_blob = next(iter(net.outputs))

exec_net = ie.load_network(network=net, device_name=device,num_requests=request_number)

n,c,h,w = net.input_info[input_blob].shape

in_frame = cv2.resize(image,(w,h))

in_frame = in_frame.transpose((2,0,1))

in_frame = in_frame.reshape((n,c,h,w))

res = exec_net.infer(inputs={input_blob:in_frame})
