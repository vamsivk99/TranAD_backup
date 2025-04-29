import pickle
import os
import pandas as pd
from tqdm import tqdm
import src.models as models # Use specific alias
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch
from time import time
from pathlib import Path
from pprint import pprint
# from beepy import beep

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Functions ----

def load_dataset(dataset):
	print("--- Entering load_dataset ---") # Added trace
	# Import args here to access args.entity
	from src.parser import args
	
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	
	# Determine entity prefix based on args.entity or dataset defaults
	entity_prefix = None
	if args.entity:
		entity_prefix = args.entity
		print(f"Loading specific entity: {entity_prefix}")
	else:
		# Fallback to default entity for specific datasets if no entity provided
		if dataset == 'SMD': entity_prefix = 'machine-1-1'
		if dataset == 'SMAP': entity_prefix = 'P-1'
		if dataset == 'MSL': entity_prefix = 'C-1'
		if dataset == 'UCR': entity_prefix = '136'
		if dataset == 'NAB': entity_prefix = 'ec2_request_latency_system_failure'
		# Add other dataset defaults if needed
		if entity_prefix:
			print(f"No entity specified, using default for {dataset}: {entity_prefix}")
		else:
			# If no entity specified and no default, assume plain train/test names
			print(f"No entity specified or default found for {dataset}. Assuming plain 'train/test/labels.npy' files.")
			entity_prefix = '' # Use empty string for plain filenames

	for file_type in ['train', 'test', 'labels']:
		# Construct filename using entity_prefix
		filename = f'{entity_prefix}_{file_type}.npy' if entity_prefix else f'{file_type}.npy'
		filepath = os.path.join(folder, filename)

		if not os.path.exists(filepath):
			# If the primary constructed path doesn't exist, try the plain name as a fallback
			fallback_filepath = os.path.join(folder, f'{file_type}.npy')
			if os.path.exists(fallback_filepath):
				print(f"Warning: Could not find {filepath}. Using plain '{file_type}.npy' instead.")
				filepath = fallback_filepath
			else:
				raise FileNotFoundError(f"Could not find data file: {filepath} (or fallback {fallback_filepath})")
		
		loader.append(np.load(filepath))
		print(f"Loaded {filepath}, shape: {loader[-1].shape}")

	if args.less: 
		print(f"Slicing training dataset ({loader[0].shape}) to 20%")
		loader[0] = cut_array(0.2, loader[0])
		print(f"Sliced training dataset shape: {loader[0].shape}")
		
	# Use batch_size=args.batch_size for training if defined?
	# Original code loaded everything, let's keep that for now.
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	# Ensure labels are numpy array
	labels = loader[2] if isinstance(loader[2], np.ndarray) else np.array(loader[2])
	print(f"Train/Test/Label shapes: {loader[0].shape} / {loader[1].shape} / {labels.shape}")
	print("--- Exiting load_dataset ---") # Added trace
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

# Define which models require sequential windowing
SEQ_MODELS = [
	'LSTMAutoencoder',
	'GRUAutoencoder',
	'VanillaTransformer', # Already present
	'ConvLSTMAutoencoder', 
	'ConvGRUAutoencoder', 
	'ConvLSTMAttentionAutoencoder',
	'ConvGRUAttentionAutoencoder',
	'TransformerEncoderDecoder',
	'LSTM_AD', 
	'MSCRED', # Already present
	'MTAD_GAT', # Already present (though potentially commented out in models.py)
	'GDN' # Already present (though potentially commented out in models.py)
]

def convert_to_windows(data, model_name, window_size=None):
	windows = []; targets = []; w_size = window_size
	data_tensor = data if isinstance(data, torch.Tensor) else torch.from_numpy(data).double()

	for i in range(len(data_tensor)):
		if i >= w_size:
			w = data_tensor[i-w_size:i]
		else:
			padding = data_tensor[0].unsqueeze(0).repeat(w_size-i-1, 1)
			data_segment = data_tensor[0:i+1]
			w = torch.cat([padding, data_segment], dim=0)

		# Target determination logic removed from here

		is_seq_model = any(m_name in model_name for m_name in SEQ_MODELS)
		
		if is_seq_model:
			windows.append(w) # Keep shape (window, features)
			targets.append(w) # Target is the window itself for reconstruction AEs
		else:
			flattened_window = w.reshape(-1)
			windows.append(flattened_window)
			targets.append(flattened_window) # Target is the flattened window

	return torch.stack(windows).double(), torch.stack(targets).double()

def load_model(modelname, dims):
	# Import constants needed for model instantiation *inside* the function
	from src.constants import n_window, lr, batch as batch_size # Rename batch to avoid conflict
	# Need to ensure constants can access args -> This might still fail if constants imports args
	# Safer approach: Pass constants explicitly from main block
	
	# --- Let's revert the import and pass args explicitly --- 
	pass # Placeholder for revised logic below

def load_model_revised(modelname, dims, n_window_val, lr_val, batch_size_val):
	print(f"--- Entering load_model_revised ({modelname}) ---") # Added trace
	model_class = getattr(models, modelname)
	
	# Prepare arguments for model constructor
	model_args = {
		'feats': dims,
		'n_window': n_window_val,
		'lr': lr_val,
		'batch_size': batch_size_val
	}
	
	try:
		model = model_class(**model_args).double()
	except TypeError as e:
		print(f"Warning: Could not instantiate {modelname} with standard args: {e}")
		print(f"Attempting instantiation with only 'feats' argument...")
		try:
			model = model_class(feats=dims).double()
			# Manually set required attributes if possible/needed after instantiation
			if hasattr(model, 'n_window'): model.n_window = n_window_val
			if hasattr(model, 'lr'): model.lr = lr_val
			if hasattr(model, 'batch'): model.batch = batch_size_val
		except Exception as fallback_e:
			print(f"Fallback instantiation failed: {fallback_e}")
			raise fallback_e

	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	epoch = -1
	accuracy_list = []
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"Loading pre-trained model: {model.name} from {fname}")
		try:
			checkpoint = torch.load(fname, map_location=device)
			model.load_state_dict(checkpoint['model_state_dict'])
			if 'optimizer_state_dict' in checkpoint and not args.test:
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			if 'scheduler_state_dict' in checkpoint and not args.test:
				scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			epoch = checkpoint.get('epoch', -1)
			accuracy_list = checkpoint.get('accuracy_list', [])
			print(f"Loaded checkpoint from epoch {epoch}")
		except Exception as e:
			print(f"Error loading checkpoint from {fname}: {e}. Creating new model.")
			# Reset epoch and accuracy list if loading fails
			epoch = -1
			accuracy_list = []
	else:
		# Message for creating new model (or if retraining)
		reason = "retraining flag set" if args.retrain else "checkpoint not found or testing without retraining"
		print(f"Creating new model: {model.name} ({reason})")
		epoch = -1
		accuracy_list = []
		
	model.to(device)
	print(f"--- Exiting load_model_revised ({modelname}) ---") # Added trace
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, targets, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1] # Feature dimension from original data
	model_name = model.name # Get model name for specific logic

	# Determine input shape and window size based on model type and data shape
	is_seq_model = any(m_name in model_name for m_name in SEQ_MODELS)
	if is_seq_model:
		# Shape is (num_samples, window_size, features)
		if data.dim() == 3:
			w_size = data.shape[1]
			expected_input_shape = (-1, w_size, feats)
		else:
			# Handle potential edge case or error - maybe data wasn't windowed correctly?
			print(f"Warning: Sequential model {model_name} received data with unexpected dimension {data.dim()}. Assuming window size 1.")
			w_size = 1 
			expected_input_shape = (-1, 1, feats) # Or adjust as needed
	else:
		# Shape is (num_samples, window_size * features)
		if data.dim() == 2 and feats > 0:
			w_size = data.shape[1] // feats
			if data.shape[1] % feats != 0:
				print(f"Warning: Non-sequential model {model_name} data dimension {data.shape[1]} not divisible by features {feats}. Cannot reliably determine window size.")
				# Fallback or error handling needed? For now, use calculated w_size.
			expected_input_shape = (-1, w_size * feats)
		else:
			# Handle potential edge case or error
			print(f"Warning: Non-sequential model {model_name} received data with unexpected dimension {data.dim()} or zero features. Assuming window size 1.")
			w_size = 1 # Or determine a suitable fallback
			expected_input_shape = (-1, 1 * feats) # Adjust based on fallback w_size

	# --- Training Loop ---
	if training:
		model.train()
		epoch_loss = 0
		num_samples = 0
		# Use DataLoader for batching during training
		batch_size = getattr(model, 'batch', 128) # Use model's batch size or default
		# Include targets in TensorDataset and DataLoader
		dataset_train = TensorDataset(data, targets)
		loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

		for batch_data, batch_target in loader_train: # Unpack data and target
			batch_data, batch_target = batch_data.to(device), batch_target.to(device) # Send both to device
			
			optimizer.zero_grad()
			
			# --- Model-specific Forward/Loss --- 
			if 'TranAD' in model_name:
				# TranAD specific forward pass (expects window, batch, feats)
				local_bs = batch_data.shape[0]
				window = batch_data.permute(1, 0, 2) # (window, batch, feats)
				elem = window[-1, :, :].view(1, local_bs, feats) # Target last element
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / (epoch + 1)) * l(z[0], elem) + (1 - 1/(epoch + 1)) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				loss_batch = torch.mean(l1)
			elif 'USAD' in model_name:
				ae1s, ae2s, ae2ae1s = model(batch_data) # Expects flattened
				l_none = nn.MSELoss(reduction = 'none')
				n = epoch + 1
				l1 = (1 / n) * l_none(ae1s, batch_data) + (1 - 1/n) * l_none(ae2ae1s, batch_data)
				l2 = (1 / n) * l_none(ae2s, batch_data) - (1 - 1/n) * l_none(ae2ae1s, batch_data)
				loss_batch = torch.mean(l1 + l2)
			else:
				# Generic forward pass for AE models (LSTM, VanillaTransformer, etc.)
				y_pred = model(batch_data) 
				# Use batch_target for loss calculation
				loss_batch = torch.mean(l(y_pred, batch_target))

			loss_batch.backward(retain_graph=(model_name == 'TranAD'))
			optimizer.step()
			
			batch_samples = batch_data.size(0)
			epoch_loss += loss_batch.item() * batch_samples
			num_samples += batch_samples
		
		scheduler.step()
		avg_loss = epoch_loss / num_samples if num_samples > 0 else 0
		tqdm.write(f'Epoch {epoch},\tAvg Loss = {avg_loss:.6f},\tLR = {optimizer.param_groups[0]["lr"]:.6f}')
		return avg_loss, optimizer.param_groups[0]['lr']

	# --- Testing Loop ---
	else: # training == False
		model.eval()
		predictions = []
		test_losses = []
		# Use DataLoader for consistency, batch size can be larger for testing
		test_batch_size = getattr(model, 'batch', 256) 
		# Include targets in TensorDataset and DataLoader
		dataset_test = TensorDataset(data, targets)
		loader_test = DataLoader(dataset_test, batch_size=test_batch_size)
		
		with torch.no_grad():
			for batch_data, batch_target in loader_test: # Unpack data and target
				batch_data, batch_target = batch_data.to(device), batch_target.to(device) # Send both to device
				
				# --- Model-specific Forward/Loss for Testing --- 
				loss_per_sample = None
				y_pred_last = None # Changed from y_pred_last
				
				if 'TranAD' in model_name:
					local_bs = batch_data.shape[0]
					window = batch_data.permute(1, 0, 2)
					elem = window[-1, :, :].view(1, local_bs, feats)
					z = model(window, elem)
					if isinstance(z, tuple): z = z[1]
					loss_per_sample = l(z, elem)[0] # (batch, feats)
					y_pred_last = z.squeeze(0)      # (batch, feats)
				elif 'USAD' in model_name:
					ae1, ae2, ae2ae1 = model(batch_data) # Expects flattened
					l_none = nn.MSELoss(reduction = 'none')
					loss_flat = 0.1 * l_none(ae1, batch_data) + 0.9 * l_none(ae2ae1, batch_data)
					# Reshape loss based on w_size and feats determined earlier
					if w_size * feats == batch_data.shape[1]: # Check if dimensions match
						loss_unflat = loss_flat.reshape(batch_data.shape[0], w_size, feats)
						loss_per_sample = torch.mean(loss_unflat, dim=1) # (batch, feats)
						y_pred_unflat = ae1.reshape(batch_data.shape[0], w_size, feats)
						y_pred_last = y_pred_unflat[:, -1, :] # (batch, feats)
					else:
						print(f"Warning: USAD - Dimension mismatch in testing loss reshaping. Using mean loss.")
						loss_per_sample = torch.mean(loss_flat, dim=1, keepdim=True).repeat(1, feats) # Fallback mean
						y_pred_last = ae1[:, -feats:] # Fallback: take last 'feats' elements
				else:
					# Generic AE testing
					y_pred = model(batch_data) # Output shape depends on model
					
					# Calculate loss using the target
					# Ensure shapes match for loss: y_pred vs batch_target
					if y_pred.shape == batch_target.shape:
						# Use per-sample loss (reduction='none')
						loss_all_samples = l(y_pred, batch_target) # Shape (batch, window, feats) or (batch, flat_feats)
						
						# Determine loss per timestamp - typically mean error of the *last* step in the window for seq models
						if is_seq_model and loss_all_samples.dim() == 3: # (batch, window, feats)
							loss_per_sample = torch.mean(loss_all_samples[:, -1, :], dim=1) # Mean of last step -> (batch,)
							# Reshape to (batch, 1) and repeat for compatibility downstream if needed
							loss_per_sample = loss_per_sample.unsqueeze(1).repeat(1, feats)
							y_pred_last = y_pred[:, -1, :] # Last predicted step (batch, feats)
						else: # Non-sequential model or unexpected shape
							# Calculate mean loss across all features/steps for the sample
							dims_to_reduce = tuple(range(1, loss_all_samples.dim())) # Reduce over all dims except batch
							loss_per_sample = torch.mean(loss_all_samples, dim=dims_to_reduce) # -> (batch,)
							# Reshape and repeat
							loss_per_sample = loss_per_sample.unsqueeze(1).repeat(1, feats)
							# Try to get the last 'feats' elements as prediction
							y_pred_flat = y_pred.view(y_pred.shape[0], -1)
							y_pred_last = y_pred_flat[:, -feats:] if y_pred_flat.shape[1] >= feats else y_pred_flat

					else:
						# Fallback if prediction and target shapes still don't match (should be less likely now)
						print(f"Warning: Shape mismatch in generic AE test loss: y_pred {y_pred.shape}, target {batch_target.shape}. Using mean loss.")
						# Ensure tensors are compatible for flatten/mean calculation
						try:
							loss_val = torch.mean(l(y_pred.flatten(), batch_target.flatten())) # Overall mean loss
							loss_per_sample = loss_val.unsqueeze(0).repeat(batch_data.shape[0], feats) # Repeat mean loss
						except RuntimeError as e:
							print(f"Error during fallback loss calculation: {e}. Setting loss to large value.")
							loss_per_sample = torch.full((batch_data.shape[0], feats), 1e5, device=device, dtype=torch.double)
						y_pred_last = y_pred # Keep original prediction

				if loss_per_sample is not None:
					test_losses.append(loss_per_sample.cpu().numpy())
				if y_pred_last is not None:
					# Ensure y_pred_last has shape (batch, feats) before appending
					if y_pred_last.shape[0] == batch_data.shape[0] and y_pred_last.dim() > 1:
						predictions.append(y_pred_last.cpu().numpy())
					else:
						# Handle case where y_pred_last might be incorrect shape
						print(f"Warning: y_pred_last shape ({y_pred_last.shape}) incorrect, padding/skipping prediction appending.")
						# Append placeholder or skip
						placeholder = np.zeros((batch_data.shape[0], feats))
						predictions.append(placeholder)

		# Concatenate results from all batches
		test_loss_np = np.concatenate(test_losses, axis=0) if test_losses else np.empty((0, feats))
		y_pred_np = np.concatenate(predictions, axis=0) if predictions else np.empty((0, feats))
		
		return test_loss_np, y_pred_np


# --- Main execution block ---
if __name__ == '__main__':
	print("--- Entering main block ---") # Added trace
	from src.parser import args # Import args first
	print(f"--- Args parsed: {args} ---") # Added trace

	# NOW import constants, as they depend on args
	# Only import lr, get n_window and batch_size from model or defaults
	from src.constants import lr, output_folder 
	import src.models as models # Ensure models is imported
	print("--- Constants and models imported ---") # Added trace

	# Load raw data (numpy arrays)
	print("--- Calling load_dataset ---") # Added trace
	train_data_np, test_data_np, labels_np = load_dataset(args.dataset)
	print("--- Returned from load_dataset ---") # Added trace

	# Handle MERLIN model (no standard training loop)
	if args.model == 'MERLIN':
		print(f"Running MERLIN model on {args.dataset}...")
		run_merlin(test_data_np, labels_np, args.dataset) # Assuming run_merlin takes numpy
		print("MERLIN execution finished.")
		exit()

	# --- Load Model --- 
	f_dim = labels_np.shape[1] # Feature dimension
	print(f"--- Feature dimension: {f_dim} ---") # Added trace

	# Need default values for n_window and batch_size before loading the model
	# These might be model-specific. Let's use some reasonable defaults
	# Check params.json or typical values? TranAD uses 10 and 128. Let's default to that.
	default_n_window = 10
	default_batch_size = 128

	print("--- Calling load_model_revised ---") # Added trace
	model, optimizer, scheduler, start_epoch, accuracy_list = load_model_revised(
		args.model, 
		f_dim, 
		args.window if args.window else default_n_window, # Use arg or default 
		lr, # From constants.py
		args.batch if args.batch else default_batch_size # Use arg or default
	)
	print("--- Returned from load_model_revised ---") # Added trace

	# --- Prepare data --- 
	trainD_np, testD_np = next(iter(train_data_np)).numpy(), next(iter(test_data_np)).numpy()
	trainO = torch.from_numpy(trainD_np).double() 
	testO = torch.from_numpy(testD_np).double()
	print("--- Original data prepared ---") # Added trace
	
	# Convert numpy arrays to windowed tensors
	window_size = getattr(model, 'n_window', default_n_window) # Get from model or use default
	print(f"--- Calling convert_to_windows (window size: {window_size}) ---") # Added trace
	trainD, trainT = convert_to_windows(trainD_np, args.model, window_size)
	testD, testT = convert_to_windows(testD_np, args.model, window_size)
	print("--- Returned from convert_to_windows ---") # Added trace

	print(f"Original train/test shapes: {trainO.shape} / {testO.shape}")
	print(f"Windowed train/test shapes: {trainD.shape} / {testD.shape}")
	print(f"Target train/test shapes: {trainT.shape} / {testT.shape}")

	training_time = 0

	# --- Training phase ---
	if not args.test:
		print("--- Entering Training Phase ---") # Added trace
		print(f'Training {args.model} on {args.dataset}')
		num_epochs = 5 # Reduced epochs for testing
		e = start_epoch + 1; start = time()
		for e in tqdm(list(range(start_epoch + 1, start_epoch + num_epochs + 1))):
			# Pass windowed data and targets to backprop
			lossT_val, lr_val = backprop(e, model, trainD, trainT, trainO, optimizer, scheduler, training=True)
			accuracy_list.append((lossT_val, lr_val))
		training_time = time() - start
		print('Training time: ' + "{:10.4f}".format(training_time) + ' s')
		save_model(model, optimizer, scheduler, e, accuracy_list)
		if accuracy_list: # Plot only if training occurred
			plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	# --- Testing phase ---
	else:
		print("--- Entering Testing Phase ---") # Added trace
		torch.zero_grad = True
		model.eval()
		print(f'Testing {args.model} on {args.dataset} Entity: {args.entity or "Default"}')
		print("--- Calling backprop for testing ---") # Added trace
		loss, y_pred = backprop(0, model, testD, testT, testO, optimizer, scheduler, training=False)
		print("--- Returned from backprop (testing) ---") # Added trace

	# --- Plot curves ---
	if not args.test:
		print(f"Plotting results for {args.model} on {args.dataset}...")
		# Ensure labels is numpy
		labels_np = labels_np if isinstance(labels_np, np.ndarray) else labels_np.numpy()
		# Plotter expects original test data (testO), predictions (y_pred), loss, labels
		# Ensure shapes match expectations of plotter function
		plotter(f'{args.model}_{args.dataset}', testO.cpu().numpy(), y_pred, loss, labels_np)

	# --- Scores ---
	df = pd.DataFrame() # Keep for potential later use or remove if only per_feature_results is needed
	per_feature_results = [] # Initialize list to store per-feature results
	print("--- Entering Scoring Phase ---") # Added trace
	print("Calculating loss on training set for POT...")
	# Pass training data and targets to backprop for loss calculation
	lossT, _ = backprop(0, model, trainD, trainT, trainO, optimizer, scheduler, training=False)

	# Ensure lossT and loss are numpy arrays
	if isinstance(lossT, torch.Tensor): lossT = lossT.cpu().detach().numpy()
	if isinstance(loss, torch.Tensor): loss = loss.cpu().detach().numpy()
	labels_np = labels_np if isinstance(labels_np, np.ndarray) else labels_np.numpy()

	preds = []
	print(f"Shape of loss (test): {loss.shape}")
	print(f"Shape of lossT (train): {lossT.shape}")
	print(f"Shape of labels: {labels_np.shape}")

	# Ensure loss dimensions align with labels for POT evaluation
	if len(loss.shape) == 1: loss = loss.reshape(-1, 1)
	if len(lossT.shape) == 1: lossT = lossT.reshape(-1, 1)
	if len(labels_np.shape) == 1: labels_np = labels_np.reshape(-1, 1)
	
	if loss.shape[0] != labels_np.shape[0]:
		print(f"Error: Test loss length ({loss.shape[0]}) does not match labels length ({labels_np.shape[0]}). Cannot evaluate.")
		# Decide how to proceed: exit or skip evaluation?
		exit(1)

	# Align feature dimensions if necessary (e.g., if loss is single value per timestamp)
	num_features_loss = loss.shape[1]
	num_features_labels = labels_np.shape[1]

	if num_features_loss != num_features_labels:
		print(f"Warning: Loss feature dimension ({num_features_loss}) != Label feature dimension ({num_features_labels}). Using overall metrics.")
		loss = np.mean(loss, axis=1)
		lossT = np.mean(lossT, axis=1)
		labels_np = (np.sum(labels_np, axis=1) >= 1).astype(int)
		# Reshape to column vectors
		loss = loss.reshape(-1, 1)
		lossT = lossT.reshape(-1, 1)
		labels_np = labels_np.reshape(-1, 1)
		num_features_eval = 1
	else:
		num_features_eval = num_features_labels
		
	print("Evaluating thresholds and metrics...")
	for i in range(num_features_eval):
		lt = lossT[:, i]
		l = loss[:, i]
		ls = labels_np[:, i]
		result, pred = pot_eval(lt, l, ls)
		preds.append(pred)
		result['feature'] = i if num_features_eval > 1 else 'overall'
		# Append result dict to list instead of using df.append
		per_feature_results.append(result)

	# Create DataFrame from the list of results after the loop
	df = pd.DataFrame(per_feature_results)

	# Calculate overall results using mean loss and combined labels
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels_np, axis=1) >= 1).astype(int)
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	
	# Calculate Hit@ and NDCG using per-feature loss if available
	if num_features_eval > 1:
		result.update(hit_att(loss, labels_np))
		result.update(ndcg(loss, labels_np))
	else: # Cannot calculate diagnosis metrics if only overall loss used
		result['Hit@100%'] = None
		result['Hit@150%'] = None
		result['NDCG@100%'] = None
		result['NDCG@150%'] = None

	print("--- Evaluation Results ---")
	if num_features_eval > 1: print("Per-feature results:\n", df)
	print("Overall results:")
	pprint(result)

	# --- Save results to CSV --- 
	summary_res = {
		'model': args.model,
		'dataset': args.dataset,
		'f1': result.get('f1'),
		'precision': result.get('precision'),
		'recall': result.get('recall'),
		'TP': result.get('TP'),
		'FP': result.get('FP'),
		'TN': result.get('TN'),
		'FN': result.get('FN'),
		'Hit@100%': result.get('Hit@100%'),
		'Hit@150%': result.get('Hit@150%'),
		'NDCG@100%': result.get('NDCG@100%'),
		'NDCG@150%': result.get('NDCG@150%'),
		'Threshold': result.get('threshold'),
		'TrainingTime': training_time if not args.test else None
	}
	results_path = Path('results/experiment_summary.csv')
	results_path.parent.mkdir(parents=True, exist_ok=True)
	summary_df = pd.DataFrame([summary_res])
	try:
		if results_path.is_file():
			summary_df.to_csv(results_path, mode='a', header=False, index=False)
		else:
			summary_df.to_csv(results_path, mode='w', header=True, index=False)
		print(f"Results appended to {results_path}")
	except Exception as e:
		print(f"Error saving results to CSV: {e}")

	# beep(4)

	print("--- Script End ---") # Added trace
