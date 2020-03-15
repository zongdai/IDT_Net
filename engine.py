import torch
import torch.nn as nn
import numpy as np
import cv2
import random

def get_patches(source, target):
	source_patches = []
	target_patches = []
	for i in range(5):
		for j in range(19):
			item = source[:, i*64:(i+1)*64, j*64:(j+1)*64]
			source.append(item)

	source_patches = torch.cat(source_patches, 0)

	for i in range(95):
		x = random.randint(0, 2048 - 64 -1)
		y = random.randint(0, 1024 - 64 -1)
		target_patches.append(target[:, y:y+64, x:x+64])
	target_patches = torch.cat(target_patches, 0)
	return source_patches, target_patches

def train(model, dataloader, num_epochs, optimizerD, optimizerG, device):
	G_losses = []
	D_losses = []
	iters = 0
	criterion = nn.BCELoss()
	l1 = nn.SmoothL1Loss()
	for epoch in range(num_epochs):
		# For each batch in the dataloader
		for i, data in enumerate(dataloader, 0):

			############################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
			## Train with all-target batch
			model.Discriminator.zero_grad()
  			# Format batch
			# source, target = get_patches(data['source'][0], data['target'])
			source = data['source'][0].to(device)
			target = data['target'][0].to(device)
			source_size = source.size(0)
			target_size = target.size(0)
			target_label = torch.full((target_size,), 1, device=device)
			# Forward pass target batch through D
			target_logit = model.Discriminator(target).view(-1)
			# Calculate loss on all-target batch
			errD_target = criterion(target_logit, target_label)
			# Calculate gradients for D in backward pass
			errD_target.backward()
			D_x = target_logit.mean().item()


			## Train with all-Synthesis batch
			
			# Generate synthesis image batch with G
			synthesis = model.MaskGenerator(source, target)
			source_label = torch.full((source_size,), 0, device=device)
			# Classify all fake batch with D
			synthesis_logits = model.Discriminator(synthesis.detach()).view(-1)
			# Calculate D's loss on the all-fake batch
			errD_synthesis = criterion(synthesis_logits, source_label)
			# Calculate the gradients for this batch
			errD_synthesis.backward()
			D_G_z1 = synthesis_logits.mean().item()
			# Add the gradients from the all-real and all-fake batches
			errD = errD_target + errD_synthesis
			# Update D
			if epoch % 2 == 0:

				optimizerD.step()


			############################
			# (2) Update G network: maximize log(D(G(z))) + L2(feature_vector)
			###########################
			model.MaskGenerator.zero_grad()
			# model.Backbone.zero_grad()
			
			
			source_label.fill_(1)  # fake labels are real for generator cost
			# Since we just updated D, perform another forward pass of all-fake batch through D
			synthesis_logits = model.Discriminator(synthesis).view(-1)
			# Calculate G's loss based on this output
			errG = criterion(synthesis_logits, source_label) 
			# Calculate gradients for G
			errG.backward()
			D_G_z2 = synthesis_logits.mean().item()
			# Update G
			optimizerG.step()

			model.MaskGenerator.zero_grad()
			synthesis = model.MaskGenerator(source, target)
			# Calculate source feature and synthesis feature
			source_feature = model.Backbone(source)
			synthesis_feature = model.Backbone(synthesis)
			errL1 = l1(source_feature, synthesis_feature) * 5
			errL1.backward()
			optimizerG.step()
			# Output training stats
			if i % 50 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_L2: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(),errL1.item(), D_x, D_G_z1, D_G_z2))
			
			# Save Losses for plotting later
			G_losses.append(errG.item())
			D_losses.append(errD.item())
			
			# Check how the generator is doing by saving G's output on fixed_noise
			if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
				imgs = synthesis.detach().cpu().numpy().transpose(0, 2, 3, 1)
				source_imgs = source.detach().cpu().numpy().transpose(0, 2, 3, 1)
				target_imgs = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
				bg = np.zeros((375, 1242, 3))
				mean = np.array([0.485, 0.456, 0.406])
				std = np.array([0.229, 0.224, 0.225])
				# for k in range(5):
				# 	path = ('../IDT_Net_Data/result/' + str(iters) + '_' + str(k) + '.png')
				# 	path_s = ('../IDT_Net_Data/result/' + str(iters) + '_' + str(k) + '_s.png')
				# 	path_t = ('../IDT_Net_Data/result/' + str(iters) + '_' + str(k) + '_t.png')
				# 	img = imgs[k, :, :, :]
				# 	img = (std * img + mean) * 255
				# 	img = np.clip(img, 0, 255)
				# 	img = img[...,[2,1,0]]
				# 	cv2.imwrite(path, img)
				# 	img = source_imgs[k, :, :, :]
				# 	img = (std * img + mean) * 255
				# 	img = np.clip(img, 0, 255)
				# 	# img = img.astype(np.int8)
				# 	img = img[...,[2,1,0]]
				# 	cv2.imwrite(path_s, img)
				# 	img = target_imgs[k, :, :, :]
				# 	img = (std * img + mean) * 255
				# 	img = np.clip(img, 0, 255)
				# 	# img = img.astype(np.int8)
				# 	img = img[...,[2,1,0]]
				# 	cv2.imwrite(path_t, img)
				for ii in range(5):
					for jj in range(19):
						bg[ii*64:(ii+1)*64, jj*64:(jj+1)*64, :] = imgs[19*ii+jj, :, :, :]
				bg = (std * bg + mean) * 255
				bg = np.clip(bg, 0, 255)
				bg = bg[...,[2,1,0]]
				cv2.imwrite('../IDT_Net_Data/result/' + str(iters) + '.png', bg)
			iters += 1