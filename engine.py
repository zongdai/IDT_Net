import torch
import torch.nn as nn

def train(model, dataloader, num_epochs, optimizerD, optimizerG, device):
	G_losses = []
	D_losses = []
	iters = 0
	criterion = nn.BCELoss()
	l2 = nn.MSELoss()
	for epoch in range(num_epochs):
    	# For each batch in the dataloader
    	for i, data in enumerate(dataloader, 0):

    		############################
	        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
	        ###########################
    		## Train with all-target batch
    		model.Discriminator.zero_grad()
  			# Format batch
    		source = data['source'].to(device)
    		target = data['target'].to(device)
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
	        synthesis = model.MaskGenerator(source)
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
	        optimizerD.step()


	        ############################
	        # (2) Update G network: maximize log(D(G(z))) + L2(feature_vector)
	        ###########################
	        model.MaskGenerator.zero_grad()
	        model.Backbone.zero_grad()
	        # Calculate source feature and synthesis feature
	        source_feature = model.Backbone(source)
	        synthesis_feature = model.Backbone(synthesis)

	        source_label.fill_(1)  # fake labels are real for generator cost
	        # Since we just updated D, perform another forward pass of all-fake batch through D
	        synthesis_logits = model.Discriminator(synthesis).view(-1)
	        # Calculate G's loss based on this output
	        errG = criterion(synthesis_logits, source_label) + l2(source_feature, synthesis_feature)
	        # Calculate gradients for G
	        errG.backward()
	        D_G_z2 = synthesis_logits.mean().item()
	        # Update G
	        optimizerG.step()



	        # Output training stats
	        if i % 50 == 0:
	            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
	                  % (epoch, num_epochs, i, len(dataloader),
	                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
	        
	        # Save Losses for plotting later
	        G_losses.append(errG.item())
	        D_losses.append(errD.item())
	        
	        # Check how the generator is doing by saving G's output on fixed_noise
	        # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
	        #     with torch.no_grad():
	        #         fake = netG(fixed_noise).detach().cpu()
	        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
	            
	        iters += 1