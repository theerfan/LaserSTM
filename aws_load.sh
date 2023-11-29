
# mkdir /mnt/oneterra/
# mount /dev/nvme2n1p1 /mnt/oneterra

# conda init bash
# conda activate pytorch
# pip install -r requirements.txt

# Set Git username
git config --global user.name "Erfan Abedi"

# Prompt for email and set it
echo "Enter your Git email:"
read git_email
git config --global user.email "$git_email"
