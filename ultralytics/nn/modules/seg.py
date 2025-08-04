import torch
import torch.nn as nn
import torch.nn.functional as F

class MulfeatSeg(nn.Module):
	def __init__(self, in_channels, height, width,n_classes =2,scale = 8):
		super().__init__()

		self.height = height
		self.width = width
		self.scale = scale

		self.cbr = conv2DBatchNormRelu(
			in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
		self.dropout = nn.Dropout2d(p=0.1)
		self.classification = nn.Conv2d(
			in_channels=128, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		for i in range(len(x)):
			x[i] = x[i].mean(0)
		# 	print(f"x = {x[i].shape}")
		out1 = F.interpolate(x[0], size=(
			self.height//self.scale, self.width//self.scale), mode="bilinear", align_corners=True)
		out2 = F.interpolate(x[1], size=(
			self.height//self.scale, self.width//self.scale), mode="bilinear", align_corners=True)
		out3 = F.interpolate(x[2], size=(
			self.height//self.scale, self.width//self.scale), mode="bilinear", align_corners=True)
		
		if (out3.size(2) != out1.size(2) or out1.size(2) != out2.size(2)):
			print(out1.size(), out2.size(), out3.size())
		 
		x = torch.cat([out1, out2, out3], dim=1)

		x = self.cbr(x)
		x = self.dropout(x)
		x = self.classification(x)
		output = F.interpolate(
			x, size=(self.height, self.width), mode="bilinear", align_corners=True)
		
		# print(f"output = {output.shape}")
		return output


class PSPNet(nn.Module):
	def __init__(self, n_classes):
		super().__init__()

		block_config = [3, 4, 6, 3]  # resnet50
		img_size = 475
		img_size_8 = 60  # img_size to 1/8

		# 4 module for subnetwork
		self.feature_conv = FeatureMap_convolution()
		self.feature_res_1 = ResidualBlockPSP(
			n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
		self.feature_res_2 = ResidualBlockPSP(
			n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1)
		self.feature_dilated_res_1 = ResidualBlockPSP(
			n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2)
		self.feature_dilated_res_2 = ResidualBlockPSP(
			n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4)

		self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[
			6, 3, 2, 1], height=img_size_8, width=img_size_8)

		self.decode_feature = DecodePSPFeature(
			height=img_size, width=img_size, n_classes=n_classes)

		self.aux = AuxiliaryPSPlayers(
			in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

	def forward(self, x):
		x = self.feature_conv(x)
		x = self.feature_res_1(x)
		x = self.feature_res_2(x)
		x = self.feature_dilated_res_1(x)

		output_aux = self.aux(x)  #  a part of Feature module into Aux module

		x = self.feature_dilated_res_2(x)

		x = self.pyramid_pooling(x)
		output = self.decode_feature(x)

		return (output, output_aux)

class conv2DBatchNormRelu(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels,
							  kernel_size, stride, padding, dilation, bias=bias)
		self.batchnorm = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		# inplace=True, decrease memory consumption by not preserving input  
		
	def forward(self, x):
		x = self.conv(x)
		x = self.batchnorm(x)
		outputs = self.relu(x)

		return outputs

class FeatureMap_convolution(nn.Module):
	def __init__(self):
		super().__init__()

		in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
		self.cbnr_1 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

		in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
		self.cbnr_2 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

		in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
		self.cbnr_3 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		x = self.cbnr_1(x)
		x = self.cbnr_2(x)
		x = self.cbnr_3(x)
		outputs = self.maxpool(x)
		return outputs

class ResidualBlockPSP(nn.Sequential):
	def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
		super().__init__()

		# bottleNeckPSP
		self.add_module(
			"block1",
			bottleNeckPSP(in_channels, mid_channels,
						  out_channels, stride, dilation)
		)

		# bottleNeckIdentifyPSP
		for i in range(n_blocks - 1):
			self.add_module(
				"block" + str(i+2),
				bottleNeckIdentifyPSP(
					out_channels, mid_channels, stride, dilation)
			)

class conv2DBatchNorm(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels,
							  kernel_size, stride, padding, dilation, bias=bias)
		self.batchnorm = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		outputs = self.batchnorm(x)

		return outputs

class bottleNeckPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
		super().__init__()

		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.cbr_2 = conv2DBatchNormRelu(
			mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
		self.cb_3 = conv2DBatchNorm(
			mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		# skip concatenate
		self.cb_residual = conv2DBatchNorm(
			in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
		residual = self.cb_residual(x)
		return self.relu(conv + residual)

class bottleNeckIdentifyPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, stride, dilation):
		super().__init__()

		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.cbr_2 = conv2DBatchNormRelu(
			mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
		self.cb_3 = conv2DBatchNorm(
			mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
		residual = x
		return self.relu(conv + residual)

class PyramidPooling(nn.Module):
	def __init__(self, in_channels, pool_sizes, height, width):
		super().__init__()

		self.height = height
		self.width = width

		out_channels = int(in_channels / len(pool_sizes))
		print(f"pool_sizes = {pool_sizes}")

		# pool_sizes: [6, 3, 2, 1]
		self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
		self.cbr_2 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
		self.cbr_3 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
		self.cbr_4 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

	def forward(self, x):
		# print(f"input = {x.shape}")
		out1 = self.cbr_1(self.avpool_1(x))
		# print(f"out1 = {out1.shape}")
		out1 = F.interpolate(out1, size=(
			self.height, self.width), mode="bilinear", align_corners=True)
		# print(f"out1 = {out1.shape}")

		out2 = self.cbr_2(self.avpool_2(x))
		# print(f"out2 = {out2.shape}")
		out2 = F.interpolate(out2, size=(
			self.height, self.width), mode="bilinear", align_corners=True)
		# print(f"out2 = {out1.shape}")

		out3 = self.cbr_3(self.avpool_3(x))
		# print(f"out3 = {out3.shape}")
		out3 = F.interpolate(out3, size=(
			self.height, self.width), mode="bilinear", align_corners=True)
		# print(f"out3 = {out3.shape}")

		out4 = self.cbr_4(self.avpool_4(x))
		# print(f"out4 = {out4.shape}")
		out4 = F.interpolate(out4, size=(
			self.height, self.width), mode="bilinear", align_corners=True)
		# print(f"out4 = {out4.shape}")
		if (x.size(2) != out1.size(2) or out1.size(2) != out2.size(2)):
			print(x.size(), out1.size(), out2.size(), out3.size(), out4.size())
		 
		output = torch.cat([x, out1, out2, out3, out4], dim=1)
		# print(f"output = {output.shape}")
		return output

class DecodePSPFeature(nn.Module):
	def __init__(self, height, width, n_classes):
		super().__init__()

		self.height = height
		self.width = width

		self.cbr = conv2DBatchNormRelu(
			in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
		self.dropout = nn.Dropout2d(p=0.1)
		self.classification = nn.Conv2d(
			in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = self.cbr(x)
		x = self.dropout(x)
		x = self.classification(x)
		output = F.interpolate(
			x, size=(self.height, self.width), mode="bilinear", align_corners=True)
		return output

class AuxiliaryPSPlayers(nn.Module):
	def __init__(self, in_channels, height, width, n_classes):
		super().__init__()

		self.height = height
		self.width = width

		self.cbr = conv2DBatchNormRelu(
			in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
		self.dropout = nn.Dropout2d(p=0.1)
		self.classification = nn.Conv2d(
			in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = self.cbr(x)
		x = self.dropout(x)
		x = self.classification(x)
		output = F.interpolate(
			x, size=(self.height, self.width), mode="bilinear", align_corners=True)
		return output

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = PSPNet(n_classes = 5)
	# print(model)
	cnt = 0
	for k, v in model.state_dict().items():
		# print(k, v.size(), torch.numel(v))
		cnt += torch.numel(v)
	print('total parameters:', cnt)

	batch = 2
	imgs = torch.rand((batch, 3, 475, 475), device = device)

	model.to(device)
	outputs = model(imgs)
	for i in range(len(outputs)):
		print(outputs[i].size())


