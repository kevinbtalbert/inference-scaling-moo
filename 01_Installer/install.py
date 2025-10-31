import subprocess

print(subprocess.run("cd 01_Installer && sh install.sh", shell=True))