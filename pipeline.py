import numpy as np

import synthesis as synth

import matplotlib.pyplot as plt
import matplotlib

bigfont = 24
medfont = 20
smallfont = 18

matplotlib.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', serif='Times')
plt.rc('axes', labelsize=smallfont)
plt.rc('legend', fontsize=smallfont)


## Generate synthetic image data

def loc2im(loc, n, width, noise_level, circular=False):
    # generates an n x n image with a dot at loc
    # loc = (x, y), -0.5 <= x, y <= 0.5
    # width = size of dot, between 0 and 1
    # noise_level = percent gaussian noise, between 0 and 1
    # returned image intensity normalized between 0 and 1
    x, y = loc
    if circular == 'y':
        y = (y+0.5) % 1 - 0.5
    elif circular == 'x':
        x = (x+0.5) % 1 - 0.5
    yy, xx = np.mgrid[:n,:n]/n - 0.5
    im = np.exp(-0.5*((xx - x)**2 + (yy - y)**2) / width**2)
    im += np.random.standard_normal((n,n)) * noise_level * im.max()
    if (im.max() - im.min()) > 0:
        return (im - im.min())/(im.max() - im.min())
    return im


## Running rollouts

def load_system(fname, A):
    print('loading system from', fname)
    data_dict = np.load(fname)
    nx = A.shape[0]
    controller_dict = {}
    for file in data_dict.files:
        sys = data_dict[file]
        Phi_xx, Phi_ux, Phi_xy, Phi_uy = (sys[:, :nx, :nx], sys[:, nx:, :nx], 
                                          sys[:, :nx, nx:], sys[:, nx:, nx:])
        controller_dict[file] = (Phi_xx, Phi_ux, Phi_xy, Phi_uy)
    return controller_dict


def save_system(fname, controllers):
    print('saving system to', fname)
    # Phi_xx, Phi_ux, Phi_xy, Phi_uy = sys 
    # store as block matrix of [[Phi_xx, Phi_xy], [Phi_ux, Phi_uy]]
    data_dict = dict([(name, np.block([[sys[0], sys[2]], [sys[1], sys[3]]])) \
                      for name, sys in controllers.items()])
    np.savez(fname, **data_dict)
    return True


def run_rollouts(sys_dict, A, B, C, T, x0s, ref, proc_fnc, obs_fnc):
    rollout_dict = {}
    for name, sys in sys_dict.items():
        _, Phi_ux, _, Phi_uy = sys
        x_trajs = []
        u_trajs = []
        y_trajs = []
        z_trajs = []
        for x0 in x0s:
            x, u, y, z = synth.simulate_output_feedback_loop(x0, T, A, B, C, Phi_ux, Phi_uy,
                                                             obs_fnc, proc_fnc, ref=ref)
            x_trajs.append(x)
            u_trajs.append(u)
            y_trajs.append(y)
            z_trajs.append(z)
        rollout_dict[name] = dict(xs=np.array(x_trajs), us=np.array(u_trajs),
                                  ys=np.array(y_trajs), zs=np.array(z_trajs))
    return rollout_dict


def load_rollouts(fname, skip_rollout=[]):
    print('loading rollouts from', fname)
    data_dict = np.load(fname)
    all_rollouts = dict()
    for file in data_dict.files:
        name, controller, arrname = file.split('_')
        if name in skip_rollout:
            print('skipping rollouts for', name)
            continue
        if name not in all_rollouts:
            all_rollouts[name] = dict()
        controller_dict = all_rollouts[name]
        if controller not in controller_dict:
            controller_dict[controller] = dict()
        rollout_dict = controller_dict[controller]
        rollout_dict[arrname] = data_dict[file]
    return all_rollouts


def save_rollouts(fname, all_rollouts):
    print('saving all rollouts to', fname)
    data_dict = {}
    for name, rollout_dict in all_rollouts.items():
        for controller, rollout in rollout_dict.items():
            for arrname, arr in rollout.items():
                keyname = '%s_%s_%s' % (name, controller, arrname)
                data_dict[keyname] = arr
    np.savez(fname, **data_dict)
    return True


def get_quartiles(arr):
    return (np.quantile(arr, 0.25, axis=0), 
            np.quantile(arr, 0.5, axis=0), 
            np.quantile(arr, 0.75, axis=0))


def compute_trajectory_cost(xs, us, Q, R, norm):    
    if norm.upper() == 'H2':
        norm = 2
        mult = 1 / len(xs)
    elif norm.upper() == 'L1':
        norm = np.inf
        mult = 1
    else:
        raise NotImplementedError('norm not supported')
    # xs, us are (N, state/input dim)
    us = np.vstack((us, np.zeros(us.shape[1])))
    H = np.diag(xs.dot(Q).dot(xs.T) + us.dot(R).dot(us.T))
    return np.linalg.norm(H, ord=norm) * mult


## Plotting functions


def plot_image_data(zd, zv, n_px):
    imd = np.mean(zd, axis=0).reshape((n_px, n_px))
    imv = np.mean(zv, axis=0).reshape((n_px, n_px))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imd); ax1.axis('off'); ax1.set_title('mean training image');
    ax2.imshow(imv); ax2.axis('off'); ax2.set_title('mean validation image');

    
def plot_coordinate_data(xs, labels):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True);
    for x, label in zip(xs, labels):
        ax1.plot(x[:,0], label=label);
        ax2.plot(x[:,2], label=label);
    ax1.set_ylabel('x coordinate'); ax1.legend();
    ax2.set_ylabel('y coordinate'); ax2.legend();
    
       
def plot_rollouts(rollout_dict, C, rs, figsize=(7, 3), skip_keys=[]):    
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=True);
    fig2, ax2 = plt.subplots(1, 1, figsize=figsize, tight_layout=True);
    
    for name, rollouts in rollout_dict.items():
        if name in skip_keys:
            continue
        xs, ys = rollouts['xs'], rollouts['ys']
        T = xs.shape[1]
        ref = 0 if rs is None else rs[None,:T, :]
        x_dist = np.linalg.norm((xs - ref).dot(C.T), axis=-1)
        y_dist = np.linalg.norm(ys - xs[:,:-1,:].dot(C.T), axis=-1)
        x_lo, x_med, x_hi = get_quartiles(x_dist)
        y_lo, y_med, y_hi = get_quartiles(y_dist)
        ax1.plot(x_med, label=name);
        ax1.fill_between(np.arange(T), x_lo, x_hi, alpha=0.3)
        ax2.plot(y_med, label=name)
        ax2.fill_between(np.arange(T-1), y_lo, y_hi, alpha=0.3)
    ax1.legend(); ax1.set_xlabel('time step'); ax1.set_ylabel("$\| x - r \|$"); 
    ax2.legend(); ax2.set_xlabel('time step'); ax2.set_ylabel("$\| p(z) - Cx \|$");
    return fig1, fig2


def plot_image_rollouts(fig, z1, z2, ref, n_px, start, step, end, title=''):
    n_steps = (end - start) // step
    im_ref = None if ref is None else (ref + 0.5)*n_px
    for i, k in enumerate(range(start, start + n_steps*step, step)):
        im = np.ones((n_px, n_px, 3))
        if z1 is not None:
            im1 = z1[k].flatten().reshape((n_px, n_px))
            im[:,:,0] -= im1; im[:,:,1] -= im1
        if z2 is not None:
            im2 = z2[k].flatten().reshape((n_px, n_px))
            im[:,:,1] -= im2
        im = np.maximum(np.zeros((n_px, n_px, 3)), im)

        plt.subplot(2, np.ceil(n_steps/2), i+1);
        plt.imshow(im);
        if im_ref is not None:
            plt.plot(im_ref[:,0], im_ref[:,2], linestyle=':', color='0.8');
            plt.plot(im_ref[k,0], im_ref[k,2], 'o', color='tab:orange');
        else:
            plt.plot(n_px//2, n_px//2, 'o', color='tab:orange');
        plt.axis('off');
    fig.suptitle(title);
    
    
def plot_error_im(ax, valset, C, P, title=''):
    xval, zval = valset
    ax.set_title(title)
    err_val = np.linalg.norm(xval.dot(C.T) - zval.dot(P.T), axis=1)
    n_px = int(np.sqrt(zval.shape[1]))
    err_im = np.mean(zval * err_val[:,None], axis=0)[1:].reshape((n_px, n_px))
    ax.imshow(err_im/err_im.max()); ax.axis('off');
    
    
def plot_err_vs_norm(ax, datasets, err_model, C):
    eps_c, eps_eta = err_model
    for data in datasets:
        err = np.linalg.norm(data['x'].dot(C.T) - data['y'], axis=-1)
        norm = np.linalg.norm(data['x'], axis=-1)
        ax.plot(norm, err, data['marker'], label=data['label'], color=data['color'])
        data['eps_G'] = np.max(err - eps_c*norm - eps_eta)

    x_err = ax.get_xlim()
    for data in datasets:
        y_err = [eps_c * x + eps_eta + data['eps_G'] for x in x_err]
        ax.plot(x_err, y_err, '--', color=data['color'])

    ax.set_ylabel("$\| p(z) - Cx \|$");
    ax.set_xlabel('$\| x \|$');
    if len(datasets) > 1:
        ax.legend()
    