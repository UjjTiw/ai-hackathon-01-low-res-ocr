import os

def patch_grayscale_import():
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("❌ CONDA_PREFIX is not set. Are you sure the conda env is activated?")
        return

    file_path = os.path.join(
        conda_prefix, 'lib', 'python3.10', 'site-packages',
        'basicsr', 'data', 'degradations.py'
    )

    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = [line.replace(
        'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
        'from torchvision.transforms.functional import rgb_to_grayscale'
    ) for line in lines]

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

    print("✅ Successfully patched rgb_to_grayscale import.")

if __name__ == "__main__":
    patch_grayscale_import()
