import subprocess
import datetime


def log_git_details(log_file = "dacer.diff"):
    try:
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get the git status
        status_result = subprocess.run(["git", "status"], capture_output=True, text=True, check=True)
        git_status = status_result.stdout

        # Get the git commit hash
        status_result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        git_commit_hash = status_result.stdout

        # Get the detailed diff of changes
        diff_result = subprocess.run(["git", "diff"], capture_output=True, text=True, check=True)
        git_diff = diff_result.stdout

        # Append the details to the log file
        with open(log_file, "w") as file:
            file.write(f"Timestamp: {timestamp}\n")
            file.write("=== Git Status ===\n")
            file.write(git_status)
            file.write("=== Git Commit Hash ===\n")
            file.write(git_commit_hash)
            file.write("\n=== Git Diff ===\n")
            file.write(git_diff)
            file.write("\n" + "-" * 80 + "\n")

        print(f"Git modification details logged to {log_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Git commands: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    log_git_details()
