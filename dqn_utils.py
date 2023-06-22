import numpy as np
import torch
import os
import sys
from imageio import mimsave
import cv2

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)

def seed_everything(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def handle_step(random_state, cnt, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, replay_buffer, checkpoint='', n_ensemble=1, bernoulli_p=1.0):
    # mask,确定哪个head可以使用本次的经验
    exp_mask = random_state.binomial(1, bernoulli_p, n_ensemble).astype(np.uint8)
    experience =  [S_prime, action, reward, finished, exp_mask, k_used, acts, cnt]
    batch = replay_buffer.send((checkpoint, experience))
    # 更新，使“状态”表示形式超过history_size帧
    S_hist.pop(0)
    S_hist.append(S_prime)
    episodic_reward += reward
    cnt+=1
    return cnt, S_hist, batch, episodic_reward

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """ epsilon线性衰减
    输入:
      decay_period: float, 衰减的周期.
      step: int, 完成的训练步数.
      warmup_steps: int, 衰减前所经过的步数.
      epsilon: float, 参数衰减的最终值.
    Returns:
      A float, 根据时间表计算出的当前epsilon值.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus

def write_info_file(info, model_base_filepath, cnt):
    info_filename = model_base_filepath + "_%010d_info.txt"%cnt
    info_f = open(info_filename, 'w')
    for (key,val) in info.items():
        info_f.write('%s=%s\n'%(key,val))
    info_f.close()

def generate_gif(base_dir, step_number, frames_for_gif, reward, name='', results=[]):
    """
        输入:
            step_number: Integer, 确定当前帧的编号
            frames_for_gif: RGB格式的Atari游戏的(210,160,3)帧序列
            reward: Integer, 总回报，输出为gif
            path: String, gif保存的路径
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = cv2.resize(frame_idx, (320, 220)).astype(np.uint8)

    if len(frames_for_gif[0].shape) == 2:
        name+='gray'
    else:
        name+='color'
    gif_fname = os.path.join(base_dir, "ATARI_step%010d_r%04d_%s.gif"%(step_number, int(reward), name))

    print("WRITING GIF", gif_fname)
    mimsave(gif_fname, frames_for_gif, duration=1/30)
