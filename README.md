# The Student Cluster Guide

[`Digital Humans 2024`](https://vlg.inf.ethz.ch/teaching/Digital-Humans-FS-24.html) | `Tutorial 2` | `29.2.2024`

ETH Zurich's Department of Computer Science (D-INFK) offers a dedicated student cluster equipped with GPUs for teaching purposes. Access to this cluster is exclusively provided to students enrolled in specific courses and their teaching assistants. Detailed guidance for utilizing the cluster is available in the help section [here](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster).

This tutorial aims to introduce a simple workflow for engaging with the cluster, covering aspects from logging in to debugging Python code in PyCharm, and orchestrating a series of experiments. It's important to note that while many workflows exist for cluster usage, this tutorial outlines one of the many approaches, specifically the one I commonly employ.

**Note**: Our course does not utilize the Euler cluster, which was previously used. The Euler student share will remain active only until March 2024, thus we have transitioned to using the D-INFK student cluster. Nevertheless, the workflow for both clusters remains largely similar.

## Tutorial Outline

1. **[Logging into the Cluster](#1-logging-into-the-cluster)**
2. **[Useful Configuration Files](#2-useful-configuration-files)**
3. **[Python Environment Setup](#3-python-environment-setup)**
4. **[Accessing GPUs: Running a Job Using `srun`](#4-accessing-gpus-running-a-job-using-srun)**
5. **[Accessing GPUs: Running a Batch of Jobs Using `sbatch`](#5-accessing-gpus-running-a-batch-of-jobs-using-sbatch)**
6. **[Accessing GPUs: Monitoring Jobs](#6-accessing-gpus-monitoring-jobs)**
7. **[Storage Options](#7-storage-options)**
8. **[Useful Command Tools](#8-useful-command-tools)**
9. **[Remote File Access](#9-remote-file-access)**
10. **[Development Setup](#10-development-setup)**

## 1. Logging into the Cluster

To log in for the first time, execute the following command in your favorite terminal emulator, replacing `login_name` with your ETH username (the one in your `login_name@ethz.ch` email):

```bash
ssh login_name@student-cluster.inf.ethz.ch
# password: ... <-- enter your ETH email password
# Last login: ...
```

This command connects you to a "login" node. These nodes, lacking GPUs, are intended for job scheduling and management rather than computation. **Avoid running intensive processes on login nodes** to prevent potential access restrictions.

For ease of access and enhanced security, we recommend setting up SSH Keys. Linux users can find setup instructions [here](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-20-04). Successfully setting up the keys will eliminate the need for password entry during subsequent logins.

```bash
ssh login_name@student-cluster.inf.ethz.ch
# Last login: ...
```

Access from outside the ETH network requires a VPN connection. Refer to the VPN setup instructions [here](https://www.isg.inf.ethz.ch/Main/ServicesNetworkVPN). On Linux, connecting is as simple as:

1. `sudo openconnect sslvpn.ethz.ch --user login_name@student-net.ethz.ch`
2. Select `student-net`.
3. Enter your ETH Wi-Fi password.
4. Enter your 6-digit multifactor authentication key, setup instructions found [here](https://ethz.ch/staffnet/en/it-services/catalogue/identity-access/multifactor-authentication.html).

## 2. Useful Configuration Files

Consider updating your `~/.bashrc` file for your convenience and needs. For example, you can adapt the [`.bashrc`](./.bashrc) file we provide. If using tmux, consider using the provided configuration file [`.tmux.conf`](./.tmux.conf) that allows you to use the mouse and simplifies the shortcuts.

## 3. Python Environment Setup

You may use either conda or venv for setting up your Python environment. To install conda, follow these steps:

```bash
# 1. Download and install conda
cd ~/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# - Accept the license agreement
# - Use your home directory as the installation location
# - Allow the installer to update your shell profile for conda initialization

# 2. Reactivate your shell for the changes to take effect
exit
ssh login_name@student-cluster.inf.ethz.ch

# 3. Create and activate a new conda environment
conda create -n dummy-env python=3.9 -y
conda activate dummy-env

# 4. Install PyTorch (adjust version numbers as needed per https://pytorch.org/get-started/previous-versions/)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# Note: GPUs are not available on login nodes
python -c "import torch; print('Cuda available?', torch.cuda.is_available())"
# Expected output: False
# GPU nodes will return `True` when queried

# 6. Install additional packages
pip install --upgrade pip
pip install tqdm  # and others as needed
```

## 4. Accessing GPUs: Running a Job Using `srun`

Each team is entitled to one GPU, with only one user able to utilize it at a time. For this tutorial, students are grouped into random 4-member teams, with final team assignments updated by March 5th.

Available hardware specifications and usage limits are detailed [here](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster). Each team is provided with 8 weeks of GPU runtime, and a job's maximum duration is capped at 2 days. Upon exceeding these limits, jobs are automatically terminated. If your project requires additional resources, please contact Frano.

To initiate a GPU session, use the following command:

```bash
srun --account digital_humans --time=00:00:30 --gpus=1 --pty bash
```

This command requests a GPU session with a specified time limit of 30 seconds (HH:MM:SS). Once resources are allocated, you'll gain access to an interactive shell on a GPU node. Note that the hostname in your shell prompt should now have been changed, for example from `login_name@student-cluster:~` to `login_name@studgpu-node01:~`. This means that you are now in an interactive shell session on the `studgpu-node01` machine.

Note that closing the original ssh session to the student cluster will terminate the job unless you are running `srun` in the background on the student cluster, e.g. in `tmux` or `screen`. To connect to the student cluster and automatically start a tmux session, consider using this SSH command from your local machine instead:

```bash
ssh login_name@student-cluster -t '(tmux at -t foobar || tmux new -s foobar)'
```

More examples of using `srun`:

```bash
# Ask for a 24h job limit# Request a 24-hour job limit
srun --account digital_humans --time=24:00:00 --gpus=1 --pty bash

# Request the maximum allowed 48-hour job limit
# Allocation may be delayed if the cluster is busy
srun --account digital_humans --time=48:00:00 --gpus=1 --pty bash

# Request the maximum allowed 2 CPUs
srun --account digital_humans --cpus-per-task 2 --pty bash

# Request 14GB RAM per CPU, totaling 28GB, the maximum allowed
srun --account digital_humans --cpus-per-task 2 --mem-per-cpu=14G --pty bash

# Generally, you can ask SLURM for any specific resource configurations
# including what kind of GPU, how much memory the GPU should have,
# how much RAM you need, how many CPU cores, etc.
# However, note that the student cluster only offers the GPUs described at
# https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster
# and has limitations on the number of CPUs and RAM which in our case are:
#  - Maximum of 2 CPUs per job
#  - Maximum of 28GB RAM in total
#  - One job at a time per team
# Limits can be verified using cri_show_assoc and cri_show_qos functions provided in the .bashrc example.
```

## 5. Accessing GPUs: Running a Batch of Jobs Using `sbatch`

For running a batch of jobs, we recommend using `sbatch` instead of `srun`. Basic instructions and examples are available [here](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs), with further details found in the [official documentation](https://slurm.schedmd.com/sbatch.html) and [ETH's scientific computing wiki](https://scicomp.ethz.ch/wiki/Using_the_batch_system).

We provide an example sbatch script at [`sbatch_example_script.sh`](./sbatch_example_script.sh), which you can submit by running:

```bash
sbatch sbatch_example_script.sh
# Submitted batch job 2736

cat ~/slurm_output__sbatch_example_script.sh-2736.out
# ...
# + python -c 'import torch; print('\''Cuda available?'\'', torch.cuda.is_available())'
# Cuda available? True
# + python -c 'import torch; torch.manual_seed(72); print(torch.randn((3,3)))'
# tensor([[-1.0001, -0.7250, -0.3560],
#         [-0.2706,  2.1503,  0.4779],
#         [-2.9557, -0.3567, -0.3766]])
# + echo Done.
# ...
```

## 6. Accessing GPUs: Monitoring Jobs

To oversee your jobs within the Slurm scheduling queue, use `squeue`. Jobs listed are either in execution or pending execution. Waiting jobs are accompanied by a reason for their delayed start.

```bash
# Basic squeue usage
squeue

# Display only your jobs
squeue --user login_name

# Filter jobs related to the course
squeue --account digital_humans

# More detailed squeue output
squeue -o "%8i %12j %15a %15g %10v %15b %5D %12u %4t %30R %9Q %5c %5C %8m %20b %5D %11M %11l %11L %20V %20S"
```

To cancel a job, use its JOBID found via `squeue` and run `scancel JOBID` (e.g., `scancel 48105793`). To cancel all your jobs, execute `scancel -u login_name`. For further job management options, explore [`scontrol`](https://slurm.schedmd.com/scontrol.html).

## 7. Storage Options

Every individual user is provided 10GB of disk space in their home folder `/home/login_name`. The usage of the space is printed after logging in to the cluster, e.g., as `Your home has 1124MB free space of 10000MB total`. You can use your home folder to setup your environment (e.g., via conda), store your code, logs, and data.

In case you need more space for downloading datasets, store large model checkpoints, etc., you can use your team's shared space in `/cluster/courses/digital_humans/datasets/team_ID`. We have reserved 1TB of shared space for the course in total. Please be considerate with your space usage, e.g., using 75GB per team should be safe and feasible. We will let you know if you use too much space, don't worry about it too much. If you need much more disk space, reach out to Frano.

**Note:** If your data or models are pulled via `git` and `git-lfs` and you do not plan to modify them or commit your modifications, you can use the `lfs-hardlink` command available on the student cluster to hard-link the LFS objects and save *half* of the space. Check the script by running `cat /usr/local/bin/lfs-hardlink` for more details.

## 8. Useful Command Tools

Consider checking some of the tools listed below for your workflow:

- `tmux` or `screen` - Continue command execution in the background even if the SSH connection is closed.
- `htop` - CPU and memory usage analysis and process management. Important for identifying CPU bottlenecks.
- `nvidia-smi` and `nvtop` - GPU usage monitoring tools. Important for identifying GPU bottlenecks.
- `ncdu` - Disk usage analyzer that helps identify what takes up the most space.
- `watch` - Execute a command repeatedly at a set interval, e.g., `watch -n 0.7 nvidia-smi` runs every 0.7 seconds.
- `scp` - Similar to cp but via SSH. It enables file and folder copying between local and remote machines via SSH, e.g., `scp login_name@student-cluster.inf.ethz.ch:/path/to/a/file /local/destination/path`.
- `which` - Inspect the full path of a command, e.g., `which python` and `which pip`.
- `vim` - The best editor.

Should a command not be available on the cluster, it may be possible to install it into your conda environment, e.g., `conda install ncdu -c conda-forge`, or alternatively, compile it from source. Note that certain tools like `nvtop` may require root privileges to install.

## 9. Remote File Access

For accessing and managing files on the cluster, such as reviewing generated results and logs, the following tools are recommended:

- [sshfs](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh) - Mounts a SSH-accessible remote location to a local directory, enabling normal access through your file explorer or applications, e.g., `sshfs login_name@student-cluster.inf.ethz.ch:/path/to/some/dir /mnt/path/on/local/machine`.
- Python's built-in [http.server](https://docs.python.org/3/library/http.server.html#http-server-security) module - Launching a simple HTTP server on the remote with `python -m http.server --directory /path/to/dir 8000` allows you to browse and view files in your local machine's browser after setting up port forwarding, e.g., `ssh -L 8000:localhost:8000 login_name@student-cluster.inf.ethz.ch`.
- [FileZilla (Client)](https://filezilla-project.org/) - Provides a graphical interface for file transfers between your local machine and remote servers.
- Command-line tools like `rsync` and `scp` for efficient file synchronization and copying.
- Integrated development environments (IDEs) such as PyCharm and VS Code offer built-in tools for remote file management.

To view TensorBoard results on your local machine's browser, you can use port forwarding:

```bash
# Log into the remote with port forwarding
ssh -L 6006:localhost:6006 login_name@student-cluster.inf.ethz.ch

# Start the TensorBoard server on port 6006
tensorboard --logdir /path/to/some/dir --port=6006

# Open http://localhost:6006/ in a browser on your local machine to access the TensorBoard
```

## 10. Development Setup

A straightforward development workflow is to code on your local machine, sync the code to the cluster login node, and execute it on the cluster GPU nodes. Your local machine is then used only for coding, and the GPU nodes on the cluster do all the computation. To set up the sync of the code, you can use PyCharm's built-in [deployment tools](https://www.jetbrains.com/help/pycharm/2023.3/creating-a-remote-server-configuration.html#overload), VS Code's [SFTP extension](https://marketplace.visualstudio.com/items?itemName=satiromarra.code-sftp), or use tools such as `rsync`. The GPU nodes and the login node share the same disk space in `/home/login_name` and `/cluster/courses/digital_humans/`, so the code only needs to be synced with the login node.

**Note:** It is not possible to SSH into the GPU nodes. Your local machine can SSH into the `student-cluster.inf.ethz.ch` login node, as depicted on the diagram below. Similarly, the GPU nodes (e.g., `studgpu-node01`) can also SSH into the login node. But neither your local machine nor the login node are permitted to SSH into the GPU node.

```txt
┌──────────────────┐      ┌───────────────────────────────┐      ┌──────────────────┐
│                  │      │                               │      │                  │
│  Local Machine   ├─────►│  student-cluster.inf.ethz.ch  │◄─────┤  studgpu-node01  │
│                  │      │                               │      │                  │
│ ┌─────────────┐  │      └─────────────┬─────────────────┘      └───┬──────────────┘
│ │ Source Code │  │                    │                            │               
│ └─────────────┘  │                    ▼                            │               
└──────────────────┘               ┌─────────────┐                   │               
                                   │  Deployed   │                   │               
                                   │             │◄──────────────────┘               
                                   │ Source Code │                                   
                                   └─────────────┘                                   
```

To debug the code executed on the GPU node, the overall simplest way available across all environments is the built-in [`breakpoint()`](https://docs.python.org/3/library/pdb.html) function (a shortcut for `import pdb; pdb.set_trace()`). This works on any machine where you have an interactive shell as follows:

1. Insert `breakpoint()` in your code where you wish to pause execution, possibly in an if statement if you are looking for a condition to be met.
2. Run your script as `python your_script.py`.
3. When the `breakpoint()` line is reached, the execution will be paused and you will have an interactive `pdb` debugger prompt.
4. Use pdb commands to inspect the state of the variables, etc. Common pdb commands include:
   - `h` for help
   - `s` to step to the next line of code
   - `p EXPRESSION` to evaluate and print an expression, e.g., to inspect the value of a variable
   - `c` to continue the code execution
   - `l` to list the code surrounding the breakpoint
   - `ll` to list the complete code of the currently evaluated function or frame
   - `q` to quit the debugger and stop the programm

For an example, refer to [`debugging_example_1.py`](./debugging_example_1.py) or execute `python debugging_example_1.py`.

**Optional:** However, you might still prefer to use an IDE for debugging for easier inspections and visualization of variables and data. For debugging inside PyCharm, the best IDE for Python by popular opinion, you can use a [Python Debug Server](https://www.jetbrains.com/help/pycharm/2023.3/remote-debugging-with-product.html#remote-debug-config). To set up a Python Debug Server in PyCharm, you can follow these steps:
1. Open a codebase in PyCharm that you want to work with and debug. For tutorial purposes, you can also git clone this repository and open it in PyCharm. 
2. Create an SFTP Deployment setup to sync the codebase opened in PyCharm to a location on your `student-cluster.inf.ethz.ch` (e.g., `/home/login_name/code/myproject01`). SFTP stands for SSH File Transfer Protocol and can be used by PyCharm to automatically sync the code changes you make in PyCharm to the remote server.
3. Make sure that you the changes in PyCharm are indeed deployed to the remote. For example, change something in the code, and use `cat` on the remote to check if the deployed file was updated.
4. Setup a Python Debug Server on a random port, e.g. 12345. Not all ports will work, you can stick to numbers beteween 10000 and 20000. Beware that some ports might already be used (possibly by other students following this tutorial). Make sure to add the necessary debugging snippet as is done in [`debugging_example_2.py`](./debugging_example_2.py).
5. Start a GPU job with `srun`, e.g. `srun --account digital_humans --time=00:05:00 --gpus=1 --pty bash`.
6. Make sure that the GPU node can SSH into the login node `student-cluster.inf.ethz.ch`. For simplicity and safety, you might want to consider setting up SSH keys again.
7. Allow the GPU node to reach your local machine at the debugging port (e.g., port 12345):
   1. Create a reverse tunnel on your local machine, `ssh -N -R 12345:localhost:12345 login_name@student-cluster.inf.ethz.ch`. This will make the requests to the port 12345 on `student-cluster.inf.ethz.ch` be tunneled to port 12345 on your local machine. To test if you can open a TCP connection to your local machine, run `nc -zv 127.0.0.1 12345` on `student-cluster.inf.ethz.ch`, this should succeed after the port forwarding has been set up and return `Connection to 127.0.0.1 12345 port [tcp/*] succeeded!`.
   2. Create a forward tunnel on your GPU node, `ssh -N -L 12345:localhost:12345 login_name@student-cluster.inf.ethz.ch`. This will make the requests to port 12345 of the GPU node be forwarded to port 12345 of `student-cluster.inf.ethz.ch`. But the requests at port 12345 `student-cluster.inf.ethz.ch` are already forwarded to the local machine with the reverse tunnel, so the requests from the GPU node will be directly forwarded to the local machine. You can test that the forwarding has been set up correctly by running `nc -zv 127.0.0.1 12345` on your GPU node, this should again return a success message.
8. Now that port 12345 of your GPU node is tunneled to port 12345 of your local machine, it is possible to attach debugging session on your GPU node to a debug server on your local machine:
   1. Start the debug server in PyCharm on your local machine.
   2. Run the python script on your GPU node as `python debugging_example_2.py`.
   3. Optionally configure path mappings for the debug server so that the source codes on the local and remote are automatically matched.

**Note:** Configuring a [remote interpreter via SSH](https://www.jetbrains.com/help/pycharm/2023.3/configuring-remote-interpreters-via-ssh.html) in PyCharm is more convenient than using the Python Debug Server. However, it is difficult to set it up cleanly as establishing an SSH connection with the GPU nodes is not permitted.

For VS Code users, you might want to consider setting up a [VS Code Server](https://code.visualstudio.com/docs/remote/vscode-server). If you want to contribute to the tutorial with more details on that, please do, contributions are welcome and encouraged through pull requests!

## Conclusion

This tutorial walked you through the most important aspects to get you up to speed with using the cluster. We outlined some of the ways and workflows to use the cluster, but many exist and feel free to find your own. It doesn't matter that much what workflow you use as long as you find it convenient to use and you can get work done.

To further assess your understanding, consider attempting the following tasks:

1. Train an image classifier on MNIST using `srun`. We have provided a minimal [`train_mnist.py`](./train_mnist.py) Python script that you can directly use. Please consider limiting your job time to 10 minutes or less to allow other students to get access to the resources during the tutorial session. Are the resulting test accuracies reasonable? Is the GPU utilized? Where is the computational bottleneck?
2. Train an image classifier on MNIST using `sbatch`. How can you monitor the job running in the background? How much time is the job still allowed to run before before being terminated? Can you find the SLURM log file?
3. Create a dummy folder on the student cluster and download it somehow to your local machine.
4. What happens if you ask for 3 CPUs? Or 2 GPUs?
5. Run a job with `srun` in the background with `tmux`, `screen`, or some other tool. Make sure you can close your SSH connection but keep `srun` running in the background. SSH again into the login node and verify that `srun` was still running in the background and was not terminated.
6. Start an interactive session with `srun`. Can you login into the node running the job via SSH? No, you cannot, figure out why. Guess what configuration file on the system disallows it.
