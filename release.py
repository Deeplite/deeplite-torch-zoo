import argparse
import os
import pkg_resources

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('deploy_env', type=str, choices=('stage', 'release'), help='Deploy environment')
    args = parser.parse_args()
    os.system(f"pip install .")

    nversion = pkg_resources.require("deeplite-torch-zoo")[0].version
    tag = f"{nversion}-{args.deploy_env}"

    print(f"Deploying to {args.deploy_env}....")

    print(f"Deleting local tag {tag}")
    os.system(f"git tag -d {tag}")

    print(f"Deleting remote tag {tag}")
    os.system(f"git push origin :refs/tags/{tag}")    

    print(f"Tagging local tag {tag}")
    os.system(f"git tag {tag}")

    print(f"Pushing remote tag")
    os.system(f"git push --tags")

    print(f"Deployment job started. Please check travis for the status of the build.")