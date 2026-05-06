import kagglehub

path = kagglehub.competition_download(
    'ethz-cil-text-class-2026'
)

print("Path to competition files:", path)