import tomli

with open("pyproject.toml", "rb") as f:
    data = tomli.load(f)

deps = data["project"]["dependencies"]
dev_deps = data["project"]["optional-dependencies"].get("dev", [])

with open("requirements.txt", "w") as f:
    f.write("\n".join(deps))

with open("dev-requirements.txt", "w") as f:
    f.write("\n".join(dev_deps))

print("Generated requirements.txt and dev-requirements.txt")