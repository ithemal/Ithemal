Implement a CLI for starting Ithemal on an AWS EC2 instance.

## Usage

First, you must have the underlying [AWS CLI](https://aws.amazon.com/cli/) installed. You also must have an IAM access key from [here](https://console.aws.amazon.com/iam/home), saved into `~/.ssh/USER.pem` (mine is `renda.pem`, so replace `renda` with whatever your username is for the rest of this)

Next, make sure that the AWS Docker image is up-to-date with `docker/docker_build_aws.sh` (if you're experimenting with changes to the Dockerfile, use a tag other than `latest` both in that file and in `aws/aws_utils/remote_setup.sh`).

Now, to start a new Ithemal instance, run `aws/start_instance.py renda`. (check out `aws/start_instance.py --help` for the full set of flags). This script takes a while to run, as it spins up a new EC2 instance, installs and starts Docker, copies over all *git-tracked* files from your repo, and then connects you to a tmux shell inside of the remote Docker instance (so that you can persist your shell if you disconnect)

Once your Ithemal instance is running, you can copy over more files from within your Ithemal directory by running `aws/synchronize_files.py --to renda FILE_OR_DIRECTORY_1 FILE_OR_DIRECTORY_2, ...`, or you can pull them back to your local folder with `--from` instead of `--to`. This is effectively a `scp`, except that it only works on files within your Ithemal directory.

Also, note that these all point to the same AWS-hosted MySQL database, not a local one, so don't break that if other people might be using it.

To connect to the AWS Docker instance again after disconnecting, run `aws/connect_instance.py renda`.

To shut down (*after you've synchronized files back to the local directory, since they will be destroyed on exit*), use `aws/stop_instance.py renda`.

## Architecture

This is primarily a wrapper around the `aws` CLI, which automatically sets up all of the parts that you need to care about Ithemal (e.g. connects to the already-running MySQL database, sets up docker, synchronizes files, etc.), along with a little bit of extra interactivity magic to make it a bit more bearable than the awful CLI or the unnecessarily complex web UI.
