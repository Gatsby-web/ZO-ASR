import os 
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch import nn
from jiwer import wer
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import argparse

# ZOOptimizer class definition
class ZOOptimizer:
    def __init__(self, model, zo_eps=1e-3, q_rge=10, 
                 zo_random_seed=6, lr=1e-4, scheduler=None):
        """
        Initialize the ZO optimizer.
        
        Parameters:
        - model: The model to be optimized.
        - zo_eps: Scaling factor for the perturbation.
        - q_rge: Number of samples for SPSA.
        - zo_random_seed: Random seed.
        - lr: Learning rate.
        - scheduler: Learning rate scheduler.
        """
        self.model = model
        self.zo_eps = zo_eps
        self.q_rge = q_rge
        self.zo_random_seed = zo_random_seed
        self.lr = lr
        self.scheduler = scheduler

        # Collect parameters to optimize
        self.named_parameters_to_optim = [
            (name, param) for name, param in model.named_parameters() if param.requires_grad
        ]
        
        # Initialize storage variables
        self.seeds = []
        self.projected_grads = []

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb parameters using a random vector z.
        """
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.zo_eps

    def zo_forward(self, compute_loss_func, inputs):
        """
        Calculate the loss value of the current model.
        """
        self.model.eval()
        with torch.inference_mode():
            loss = compute_loss_func(self.model, inputs)
            if hasattr(compute_loss_func, 'n_gpu') and compute_loss_func.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

    def zo_step(self, compute_loss_func, inputs):
        """
        Perform n SPSA gradient estimations and return the average loss.
        """
        torch.manual_seed(self.zo_random_seed)
        seeds = []
        projected_grads = []
        loss_sum = 0.0

        for _ in range(self.q_rge):
            # Generate a new random seed
            self.zo_random_seed = np.random.randint(1000000000)
            seeds.append(self.zo_random_seed)

            # Forward perturbation θ + εz
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(compute_loss_func, inputs)

            # Backward perturbation θ - εz
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = self.zo_forward(compute_loss_func, inputs)

            # Calculate gradient estimate
            projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
            projected_grads.append(projected_grad)
            loss_sum += loss1

            # Reset perturbation θ + εz
            self.zo_perturb_parameters(scaling_factor=1)

        # Calculate average loss
        avg_loss = loss_sum / self.q_rge
        self.seeds = seeds
        self.projected_grads = projected_grads

        return avg_loss

    def zo_update(self):
        """
        Update model parameters based on estimated gradients.
        """
        learning_rate = self.lr

        for j, projected_grad in enumerate(self.projected_grads):
            torch.manual_seed(self.seeds[j])
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data - (learning_rate * projected_grad * z)

        if self.scheduler is not None:
            self.scheduler.step()

    # def reset(self):
    #     """
    #     Reset stored seeds and gradients.
    #     """
    #     self.seeds = []
    #     self.projected_grads = []

    def state_dict(self):
        """
        Return the optimizer's state dictionary.
        """
        return {
            'zo_eps': self.zo_eps,
            'q_rge': self.q_rge,
            'zo_random_seed': self.zo_random_seed,
            'lr': self.lr,
            'seeds': self.seeds,
            'projected_grads': self.projected_grads,
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
        }

    def load_state_dict(self, state_dict):
        """
        Load the optimizer's state from a state dictionary.
        """
        self.zo_eps = state_dict['zo_eps']
        self.q_rge = state_dict['q_rge']
        self.zo_random_seed = state_dict['zo_random_seed']
        self.lr = state_dict['lr']
        self.seeds = state_dict['seeds']
        self.projected_grads = state_dict['projected_grads']
        if self.scheduler and state_dict['scheduler_state'] is not None:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])

# Other helper functions (unchanged)

def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    else: 
        return optimizer, None

def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def mcc_loss(x, reweight=False, dim=2, class_num=32):
    p = x.softmax(dim) # (1, L, D)
    p = p.squeeze(0) # (L, D)
    if reweight: # (1, L, D) * (L, 1) 
        target_entropy_weight = softmax_entropy(x, dim=2).detach().squeeze(0) # instance-wise entropy (1, L, D)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
    else:    
        cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
   
    return mcc_loss

def div_loss(x, non_blank=None, L_thd=64):
    # maximize entropy of class prediction for every time-step in a utterance 
    # x (1, L, D)
    loss = 0
    x = x.squeeze(0)
    L = x.shape[0]

    if non_blank is not None: 
        cls_pred = x.mean(0)[1:] # (D, )
    else:
        cls_pred = x.mean(0) # (D, )

    loss = -softmax_entropy(cls_pred, 0)

    return loss

def collect_params(model, bias_only=False, train_feature=False, train_all=False, train_LN=True):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    trainable = []
    if bias_only:
        trainable = ['bias']
    else: 
        trainable = ['weight', 'bias']

    for nm, m in model.named_modules():
        print(nm)
        if train_LN: 
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if train_feature:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] in ['feature_extractor', 'feature_projection']:
                    for np, p in m.named_parameters():
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_all: 
            for np, p in m.named_parameters():
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")

    return params, names

def consist_loss(model, input_values, outputs):
    targets = outputs
    # noisy outputs
    model.wav2vec2.encoder.dropout.train()
    noisy_outputs = model(input_values).logits

    import json
    with open('vocab.json') as f:
        vocab = json.load(f)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False)
    predicted_ids = torch.argmax(outputs, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    target = []
    for s in transcription:
        if s == ' ':
            s = '|'
        target.append(vocab[s])

    logp = noisy_outputs.log_softmax(1).transpose(1, 0) # L,N,D
    input_len = logp.shape[0]
    tgt_len = len(target)
    loss = ctc_loss(logp, torch.tensor(target).int(), torch.tensor([input_len]), torch.tensor([tgt_len]))
    model.eval()
    return loss

def copy_model_and_optimizer(model, optimizer, scheduler):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_state = deepcopy(optimizer.state_dict())
    elif isinstance(optimizer, ZOOptimizer):
        optimizer_state = deepcopy(optimizer.state_dict())
    else:
        raise ValueError("Unknown optimizer type")
    
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            scheduler_state = deepcopy(scheduler.state_dict())
        else:
            scheduler_state = None
    else:
        scheduler_state = None

    return model_state, optimizer_state, scheduler_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state, scheduler_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.load_state_dict(optimizer_state)
    elif isinstance(optimizer, ZOOptimizer):
        optimizer.load_state_dict(optimizer_state)
    else:
        raise ValueError("Unknown optimizer type")
    
    if scheduler_state is not None and scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else: 
        return model, optimizer, None

def cal_grad(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def configure_model(model):
    """Configure model for use with tent."""
    model.requires_grad_(False)
    return model


def compute_loss(model, inputs, em_coef, reweight, temp, non_blank, div_coef):
    model.eval()
    with torch.inference_mode():
        outputs = model(inputs).logits
        loss = 0

        if em_coef > 0: 
            if non_blank:      
                e_loss = softmax_entropy(outputs / temp)[non_blank].mean()
            else: 
                e_loss = softmax_entropy(outputs / temp).mean()
            loss += e_loss * em_coef

        if 1 - em_coef > 0: 
            c_loss = mcc_loss(outputs / temp, reweight)
            loss += c_loss * (1 - em_coef)

        if div_coef > 0: 
            d_loss = div_loss(outputs, non_blank) 
            loss += d_loss * div_coef 
        
        return loss

if __name__ == '__main__':
    SAMPLE_RATE = 16000
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--div_coef', type=float, default=0.)
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/home/daniel094144/data/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--bias_only', action='store_true')
    parser.add_argument('--train_feature', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--non_blank', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./exps')
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--scheduler', default=None)
    parser.add_argument('--q_rge', type=int, default=8)

    args = parser.parse_args()
    asr = args.asr
    steps = args.steps
    episodic = args.episodic
    opt = args.opt
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    split = args.split
    lr = args.lr
    em_coef = args.em_coef
    reweight = args.reweight
    batch_size = args.batch_size
    temp =  args.temp
    non_blank = args.non_blank
    log_dir = args.log_dir
    extra_noise = args.extra_noise
    scheduler = args.scheduler
    div_coef = args.div_coef
    bias_only = args.bias_only
    train_feature = args.train_feature
    train_all = args.train_all
    q_rge = args.q_rge
    skip_short_thd = None
    train_LN = True

    exp_name = dataset_name+'_'+str(em_coef)+'_'+str(steps)+'_'+str(temp)+'_'+asr.split('/')[-1]+'_'+'non_blank'+str(non_blank)+'_noise_'+str(extra_noise)+'_rew_'+str(reweight)+'_div_'+str(div_coef)+'_bias_'+str(bias_only)+'_feat_'+str(train_feature)+'_all_'+str(train_all)+'_LN_'+str(train_LN)

    from data import load_dataset
    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)
    
    # --- Refactored Part 1: Initialization ---
    # Define steps at which to log WER
    log_steps = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Use a dictionary to store transcriptions at different steps
    transcriptions_at_steps = {step: [] for step in log_steps}

    gt_texts = []
    ori_transcriptions = []
    durations = []
    werrs = []
    # -------------------------------------------

    print('------------------------------------')
    print(f'exp: {exp_name}')
    print(f'eposidic? {episodic}')
    print(f'lr = {lr}')
    print(f'optim = {opt}')
    print(f'step = {steps}')
    print(f'em_coef = {em_coef}')
    print(f'reweight = {reweight}')
    print(f'batch size = {batch_size}')
    print(f'temperature = {temp}')
    print(f'non_blank = {str(non_blank)}')
    print(f'extra_noise = {extra_noise}')
    print(f'scheduler = {str(scheduler)}')
    print(f'div_coef = {str(div_coef)}')
    print(f'bias_only = {bias_only}')
    print(f'train_feature = {train_feature}')
    print(f'train_all = {train_all}')
    print(f'train_LN = {train_LN}')

    # Load model and processor
    local_model_path = './checkpoints/'
    processor = Wav2Vec2Processor.from_pretrained(local_model_path, sampling_rate=SAMPLE_RATE, return_attention_mask=True)
    model = Wav2Vec2ForCTC.from_pretrained(local_model_path).eval().cuda()      

    # Configure model
    model = configure_model(model)
    # params: all parameters
    # param_names: parameters to fine-tune
    params, param_names = collect_params(model, bias_only, train_feature, train_all, train_LN)

    # Initialize ZO optimizer
    zo_optimizer = ZOOptimizer(
        model=model,
        zo_eps=1e-3,          # can be adjusted as needed
        q_rge=q_rge,            # number of SPSA samples, adjust as needed
        zo_random_seed=6,  # can set a specific seed for reproducibility
        lr=lr,
        scheduler=None        # if a learning rate scheduler is needed, pass it here
    )

    # schedule is None by default
    if scheduler is not None:
        # Initialize a traditional optimizer and scheduler, then pass the scheduler to ZOOptimizer
        traditional_optimizer, sched = setup_optimizer(params, zo_optimizer, lr, scheduler=scheduler)
        zo_optimizer.scheduler = sched

    if episodic: 
        model_state, optimizer_state, scheduler_state = copy_model_and_optimizer(model, zo_optimizer, zo_optimizer.scheduler)

    print(param_names)
    count = 0

    import time
    start = time.time()
    for batch in dataset:
        lens, wavs, texts, files = batch

        inputs = processor(wavs, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.cuda()
        duration = input_values.shape[1] / SAMPLE_RATE
        durations.append(duration)
        
        if episodic: 
            model, zo_optimizer, _ = load_model_and_optimizer(model, zo_optimizer, model_state, optimizer_state, scheduler_state)
        
        # vanilla forward 
        with torch.no_grad():
            outputs = model(input_values).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        ori_transcription = processor.batch_decode(predicted_ids)
        ori_transcriptions += ori_transcription
        ori_wer = wer(list(texts), list(ori_transcription))
        print("original WER: ", ori_wer)

        if skip_short_thd is not None: 
            if outputs.shape[1] <= skip_short_thd:
                print(f'do not adapt since length is {outputs.shape[1]}')
                count += 1
                continue

        # --- Refactored Part 2: Adaptation Loop ---
        # ZO adaptation
        for i in range(steps): 
            # Perform ZO step, estimate loss
            avg_loss = zo_optimizer.zo_step(lambda m, inp: compute_loss(m, inp, em_coef, reweight, temp, non_blank, div_coef), input_values)
            
            # Perform ZO parameter update
            zo_optimizer.zo_update()

            step_num = i + 1 # Current step (1-indexed)

            # Record and print output after adaptation
            if episodic and step_num in transcriptions_at_steps: 
                with torch.no_grad():
                    outputs = model(input_values).logits
                predicted_ids = torch.argmax(outputs, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                ada_wer = wer(list(texts), list(transcription))
                print(f"adapt-{step_num} WER:  ", ada_wer)

                # Store transcription
                transcriptions_at_steps[step_num] += transcription
                
                # Special case for WERR at step 10
                if step_num == 10:
                    werr = ori_wer - ada_wer
                    werrs.append(werr)
        # -----------------------------------------------

        del input_values
        torch.cuda.empty_cache()
        gt_texts += texts

    print("asr:", asr)
    print(f'non-adapted count = {count}')
    print(f'dataset num = {len(dataset)}')
    print("original WER:", wer(gt_texts, ori_transcriptions))
    
    # --- Refactored Part 3: Final Print ---
    # Print WER at different adaptation steps
    for step_num, transcriptions in transcriptions_at_steps.items():
        if steps >= step_num:
            print(f"TTA-{step_num} WER:", wer(gt_texts, transcriptions))
    # ----------------------------------------
            
    print('------------------------------------')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with open(os.path.join(log_dir, exp_name), 'w') as f: 
        f.write(f"original WER: {wer(gt_texts, ori_transcriptions)}\n")
        
        # --- Refactored Part 4: File Write ---
        # Write WER at different adaptation steps to log file
        for step_num, transcriptions in transcriptions_at_steps.items():
            if steps >= step_num:
                f.write(f"TTA-{step_num} WER: {wer(gt_texts, transcriptions)}\n")
        # ---------------------------------------
                
        f.write(f'eposidic? {episodic}\n')
        f.write(f'lr = {lr}\n')
        f.write(f'optim = {opt}\n')
        f.write(f'step = {steps}\n')
        f.write(f'em_coef = {em_coef}\n')
        f.write(f'reweight = {reweight}\n')
        f.write(f'batch size = {batch_size}\n')
        f.write(f'temperature = {temp}\n')
        f.write(f'non_blank = {str(non_blank)}\n')
        f.write(f'extra_noise = {extra_noise}\n')
        f.write(f'scheduler = {str(scheduler)}\n')
        f.write(f'div_coef = {str(div_coef)}\n')
        f.write(f'bias_only = {str(bias_only)}\n')
        f.write(f'train_feature = {str(train_feature)}\n')
        f.write(f'train_all = {str(train_all)}\n')
        f.write(f'train_LN = {str(train_LN)}\n')
    
    csv_path = os.path.join(log_dir, exp_name+'.csv')
    df = pd.DataFrame({'duration': durations, 'WERR': werrs})
    df.to_csv(csv_path)