#!/bin/bash

# ==============================================================================
# CLEANUP SCRIPT - Educational Version with Per-Folder Progress Bars
# ==============================================================================
# PURPOSE: This script deletes specific files and folders to clean up your system.
# USAGE: Run with './cleanup.sh' or using the alias we'll create.
# FEATURES: Shows a progress bar for EACH folder being cleaned!
# ==============================================================================

# SYNTAX: '#!/bin/bash' is called a "shebang". It tells the system to use bash
# to execute this script (bash is the shell/command interpreter we're using).

# ------------------------------------------------------------------------------
# PROGRESS BAR FUNCTIONS
# ------------------------------------------------------------------------------

# CONCEPT: This function draws a visual progress bar in the terminal.
# It's like a loading bar you see in apps, but made with text characters.

# SYNTAX: Functions in bash are defined like this:
# function_name() { commands }

show_progress_bar() {
    # PARAMETERS EXPLAINED:
    # $1 = current step number
    # $2 = total number of steps
    # $3 = description text
    # $4 = prefix (optional, for indentation)

    local current=$1
    local total=$2
    local text=$3
    local prefix="${4:-}"  # Default to empty if not provided

    # CONCEPT: Prevent division by zero
    # If total is 0, set it to 1 to avoid math errors
    if [ $total -eq 0 ]; then
        total=1
    fi

    # CONCEPT: Calculate the percentage complete
    # SYNTAX: $(( )) allows us to do math in bash
    local percent=$((current * 100 / total))

    # CONCEPT: Calculate how many "filled" characters to show in the bar
    # A bar width of 40 characters (smaller than before to fit indentation)
    local filled=$((percent * 40 / 100))

    # CONCEPT: Truncate text if it's too long to prevent wrapping
    local max_text_length=35
    if [ ${#text} -gt $max_text_length ]; then
        text="${text:0:32}..."
    fi

    # CONCEPT: Build the progress bar string
    local bar=""
    local i

    for i in $(seq 0 39); do
        if [ $i -lt $filled ]; then
            bar+="█"  # Filled block character
        else
            bar+="░"  # Empty block character
        fi
    done

    # CONCEPT: '\r' moves cursor to start of line for animation effect
    # We clear the line first, then print the progress bar
    printf "\r%-120s" ""  # Clear the line
    printf "\r%s[%s] %3d%% - %s" "$prefix" "$bar" "$percent" "$text"
}

# CONCEPT: Count files in a directory recursively
# This helps us show accurate progress when deleting folders
count_files_in_dir() {
    local dir=$1

    # SYNTAX: 'find' searches for files/folders
    # -type f = only files (not directories)
    # 2>/dev/null = hide error messages (e.g., permission denied)
    # wc -l = count the number of lines (each file is one line)

    local count=$(find "$dir" -type f 2>/dev/null | wc -l)
    echo $count
}

# CONCEPT: Delete a folder with progress bar showing based on time estimation
delete_folder_with_progress() {
    local folder_path=$1
    local folder_name=$(basename "$folder_path")

    echo ""
    echo "📂 Processing: $folder_path"

    # CONCEPT: Check if the folder exists
    if [ ! -e "$folder_path" ]; then
        echo "   ⏭️  Skipped: Folder doesn't exist"
        return
    fi

    # CONCEPT: Check if it's actually a file, not a folder
    if [ -f "$folder_path" ]; then
        # It's a single file, not a folder
        rm -f "$folder_path" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "   ✅ Deleted: Single file"
        else
            echo "   ❌ Failed: Could not delete file"
        fi
        return
    fi

    # CONCEPT: Count total files for display (but we won't delete one-by-one!)
    echo "   🔍 Counting files..."
    local total_files=$(count_files_in_dir "$folder_path")

    if [ $total_files -eq 0 ]; then
        # Empty folder or only subdirectories
        rm -rf "$folder_path" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "   ✅ Deleted: Empty folder"
        else
            echo "   ❌ Failed: Could not delete folder"
        fi
        return
    fi

    echo "   📊 Found $total_files files to delete"
    echo "   🚀 Deleting (fast mode)..."

    # CONCEPT: PERFORMANCE OPTIMIZATION
    # Instead of deleting files one-by-one (SLOW), we:
    # 1. Start the fast batch deletion in the background
    # 2. Show a progress bar that animates based on time
    # 3. This gives visual feedback while deletion runs at full speed

    # SYNTAX: Start deletion in background
    # The '&' at the end makes it run in parallel with our progress bar
    rm -rf "$folder_path" 2>/dev/null &
    local delete_pid=$!  # Store the process ID of the background deletion

    # CONCEPT: Estimate how long deletion will take
    # Rule of thumb: ~1000 files per second on average hardware
    # You can adjust this multiplier based on your system speed
    local estimated_seconds=$(awk "BEGIN {print $total_files / 1000}")

    # SYNTAX: awk is a calculator for floating point math in bash
    # We can't use $(( )) for decimals, so we use awk
    # This gives us a rough time estimate

    # Minimum 1 second, maximum 30 seconds for progress bar animation
    if (( $(awk "BEGIN {print ($estimated_seconds < 1)}") )); then
        estimated_seconds=1
    elif (( $(awk "BEGIN {print ($estimated_seconds > 30)}") )); then
        estimated_seconds=30
    fi

    # CONCEPT: Show animated progress bar while deletion happens
    local progress=0
    local update_interval=0.1  # Update every 100ms
    local total_updates=$(awk "BEGIN {print int($estimated_seconds / $update_interval)}")

    # SYNTAX: Loop while the deletion process is still running
    # 'kill -0 $pid' checks if process exists (doesn't actually kill it)
    while kill -0 $delete_pid 2>/dev/null; do
        # Update progress
        ((progress++))

        # CONCEPT: Calculate percentage based on time elapsed
        # Not perfectly accurate, but gives good visual feedback
        local percent=$(awk "BEGIN {print int(($progress / $total_updates) * 100)}")

        # Cap at 99% until actually done
        if [ $percent -gt 99 ]; then
            percent=99
        fi

        # CONCEPT: Show progress without filename (MUCH faster!)
        # No basename calls, no string processing, just numbers
        show_progress_bar $percent 100 "Deleting..." "   "

        # Small sleep to not hammer the CPU
        sleep $update_interval
    done

    # CONCEPT: Wait for deletion to complete and get exit code
    wait $delete_pid
    local exit_code=$?

    # Show 100% completion
    show_progress_bar 100 100 "Complete!" "   "
    echo ""

    if [ $exit_code -eq 0 ]; then
        echo "   ✅ Completed: $total_files files deleted"
    else
        echo "   ⚠️  Completed with errors (some files may remain)"
    fi
}

# ------------------------------------------------------------------------------
# PYTHON CACHE CLEANUP FUNCTION
# ------------------------------------------------------------------------------

# CONCEPT: Clean Python cache files (__pycache__ and .pyc) with progress
clean_python_cache() {
    local project_dir=$1
    local project_name=$(basename "$project_dir")

    echo ""
    echo "🐍 Processing Python cache in: $project_dir"

    # CONCEPT: Check if directory exists
    if [ ! -d "$project_dir" ]; then
        echo "   ⏭️  Skipped: Directory doesn't exist"
        return
    fi

    # STEP 1: Clean __pycache__ directories (FAST MODE)
    # ========================================

    echo "   🔍 Finding __pycache__ directories..."

    # SYNTAX: Count first to show in progress
    local pycache_count=$(find "$project_dir" -type d -name "__pycache__" 2>/dev/null | wc -l)

    if [ $pycache_count -gt 0 ]; then
        echo "   📊 Found $pycache_count __pycache__ directories"
        echo "   🚀 Deleting (fast mode)..."

        # CONCEPT: FAST BATCH DELETION
        # Use the efficient -exec rm -rf {} + approach (your original command)
        # Run in background and show progress based on time

        find "$project_dir" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null &
        local delete_pid=$!

        # Estimate: ~50 directories per second
        local estimated_seconds=$(awk "BEGIN {print $pycache_count / 50}")
        if (( $(awk "BEGIN {print ($estimated_seconds < 1)}") )); then
            estimated_seconds=1
        elif (( $(awk "BEGIN {print ($estimated_seconds > 10)}") )); then
            estimated_seconds=10
        fi

        # Animate progress bar
        local progress=0
        local total_updates=$(awk "BEGIN {print int($estimated_seconds / 0.1)}")

        while kill -0 $delete_pid 2>/dev/null; do
            ((progress++))
            local percent=$(awk "BEGIN {print int(($progress / $total_updates) * 100)}")
            if [ $percent -gt 99 ]; then percent=99; fi

            show_progress_bar $percent 100 "Removing __pycache__..." "   "
            sleep 0.1
        done

        wait $delete_pid
        show_progress_bar 100 100 "Complete!" "   "

        echo ""  # Newline after progress bar
        echo "   ✅ Removed $pycache_count __pycache__ directories"
    else
        echo "   ℹ️  No __pycache__ directories found"
    fi

    # STEP 2: Clean .pyc files (ALREADY FAST)
    # ========================================

    echo "   🔍 Finding .pyc files..."

    # CONCEPT: Count files first
    local pyc_count=$(find "$project_dir" -name "*.pyc" 2>/dev/null | wc -l)

    if [ $pyc_count -gt 0 ]; then
        echo "   📊 Found $pyc_count .pyc files"
        echo "   🚀 Deleting (fast mode)..."

        # SYNTAX: 'find -delete' is already very efficient
        # Run in background with progress animation

        find "$project_dir" -name "*.pyc" -delete 2>/dev/null &
        local delete_pid=$!

        # Estimate: ~500 files per second
        local estimated_seconds=$(awk "BEGIN {print $pyc_count / 500}")
        if (( $(awk "BEGIN {print ($estimated_seconds < 1)}") )); then
            estimated_seconds=1
        elif (( $(awk "BEGIN {print ($estimated_seconds > 10)}") )); then
            estimated_seconds=10
        fi

        # Animate progress bar
        local progress=0
        local total_updates=$(awk "BEGIN {print int($estimated_seconds / 0.1)}")

        while kill -0 $delete_pid 2>/dev/null; do
            ((progress++))
            local percent=$(awk "BEGIN {print int(($progress / $total_updates) * 100)}")
            if [ $percent -gt 99 ]; then percent=99; fi

            show_progress_bar $percent 100 "Removing .pyc files..." "   "
            sleep 0.1
        done

        wait $delete_pid
        show_progress_bar 100 100 "Complete!" "   "

        echo ""
        echo "   ✅ Removed $pyc_count .pyc files"
    else
        echo "   ℹ️  No .pyc files found"
    fi
}

# ------------------------------------------------------------------------------
# SECTION 1: START MESSAGE
# ------------------------------------------------------------------------------

# CONCEPT: We inform the user that cleanup is starting.
# No confirmation needed since we're cleaning known temporary folders.

echo "🗑️  Starting cleanup of temporary files..."
echo ""

# ------------------------------------------------------------------------------
# SECTION 2: DEFINE WHAT TO DELETE
# ------------------------------------------------------------------------------

# CONCEPT: We use an array (a list) to store all paths we want to delete.
# This makes it easy to add or remove items later.

# SYNTAX: In bash, arrays are created with parentheses ( )
# Each item is separated by a space
# We use quotes around paths with spaces (though these don't have spaces)
ITEMS_TO_DELETE=(
    "$HOME/microwakeword_trainer/models"
    "$HOME/microwakeword_trainer/logs"
    "$HOME/microwakeword_trainer/data"
    "$HOME/microwakeword_trainer/exports"
    "$HOME/microwakeword_trainer/__pycache__"
    "$HOME/microwakeword_trainer/scripts/__pycache__"
    "$HOME/microwakeword_trainer/tests/__pycache__"
    "$HOME/microwakeword_trainer/tests/*/__pycache__"
    "$HOME/microwakeword_trainer/official_models/__pycache__"
    "$HOME/microwakeword_trainer/config/__pycache__"
    "$HOME/microwakeword_trainer/cache"
    "$HOME/microwakeword_trainer/profiles"
    "$HOME/microwakeword_trainer/custom_output"
    "$HOME/microwakeword_trainer/test_output"
    "$HOME/microwakeword_trainer/ci_export"
    "$HOME/microwakeword_trainer/checkpoints"
    "$HOME/microwakeword_trainer/tuning"
)

# SYNTAX: '$HOME' is a variable that contains your home directory path
# For example: /home/yourusername
# Using $HOME makes the script work for any user

# ------------------------------------------------------------------------------
# SECTION 2B: PYTHON CACHE CLEANUP CONFIGURATION
# ------------------------------------------------------------------------------

# CONCEPT: Python creates cache files (__pycache__ folders and .pyc files)
# to speed up imports. These can be safely deleted and will be recreated.

# SYNTAX: true/false - Enable or disable Python cache cleanup
CLEAN_PYTHON_CACHE=true

# CONCEPT: We define directories where we want to clean Python cache
# This is separate from ITEMS_TO_DELETE because we're searching recursively
# for specific Python cache patterns, not deleting entire folders.

PYTHON_PROJECTS=(
    "$HOME/microwakeword_trainer/src"                                    # Your source code directory
    #"."                 # Example: another project
    # "$HOME/code"                           # Example: all code folder
)

# EXPLANATION: Why separate Python cleanup?
# - __pycache__ folders can be ANYWHERE in your project tree
# - We need to find them recursively (search all subdirectories)
# - Regular folder deletion would delete your entire src/ folder!
# - This targeted approach only removes cache, keeps your code safe

# ------------------------------------------------------------------------------
# SECTION 3: DELETE ITEMS WITH PER-FOLDER PROGRESS BARS
# ------------------------------------------------------------------------------

echo ""
echo "🗑️  Starting cleanup..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# CONCEPT: We need to know the total count for the overall summary
# SYNTAX: ${#ARRAY[@]} gives us the number of items in an array
total_items=${#ITEMS_TO_DELETE[@]}
current_item=0
total_files_deleted=0

# CONCEPT: We loop through each item in our array and delete it
# SYNTAX: 'for item in "${ITEMS_TO_DELETE[@]}"; do'
#   - 'for' starts the loop
#   - 'item' is a temporary variable that holds the current path
#   - 'in' separates the variable from the list
#   - '"${ITEMS_TO_DELETE[@]}"' means "all items in the array"
#   - 'do' marks the start of the loop body
#   - 'done' marks the end of the loop

for item in "${ITEMS_TO_DELETE[@]}"; do

    # CONCEPT: Increment the current item counter
    # SYNTAX: ((variable++)) increases the variable by 1
    ((current_item++))

    echo "[$current_item/$total_items] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # CONCEPT: Use our new function that shows per-file progress
    # This will handle all the deletion logic and display progress bars
    delete_folder_with_progress "$item"

done

# ------------------------------------------------------------------------------
# SECTION 3B: PYTHON CACHE CLEANUP
# ------------------------------------------------------------------------------

# CONCEPT: After cleaning folders, clean Python cache if enabled
if [ "$CLEAN_PYTHON_CACHE" = true ]; then

    # Check if we have any Python projects defined
    if [ ${#PYTHON_PROJECTS[@]} -gt 0 ]; then

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🐍 PYTHON CACHE CLEANUP"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # CONCEPT: Loop through each Python project directory
        python_project_count=${#PYTHON_PROJECTS[@]}
        python_current=0

        for project in "${PYTHON_PROJECTS[@]}"; do
            ((python_current++))

            echo ""
            echo "[Python $python_current/$python_project_count] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            # CONCEPT: Call our Python cleanup function
            clean_python_cache "$project"
        done

    else
        echo ""
        echo "ℹ️  Python cache cleanup enabled but no projects defined"
    fi

else
    echo ""
    echo "ℹ️  Python cache cleanup disabled (set CLEAN_PYTHON_CACHE=true to enable)"
fi

# ------------------------------------------------------------------------------
# SECTION 4: COMPLETION MESSAGE
# ------------------------------------------------------------------------------

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Cleanup complete!"
echo "   Total items processed: $total_items"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# SYNTAX: 'exit 0' means "script finished successfully"
# This is important for automation - other scripts can check if this succeeded
exit 0

# ==============================================================================
# HOW TO USE THIS SCRIPT:
# ==============================================================================
# 1. Make it executable: chmod +x cleanup.sh
# 2. Run it: ./cleanup.sh
# 3. OR use the alias we'll add to .bashrc (see next step)
# ==============================================================================

# ==============================================================================
# CUSTOMIZATION GUIDE:
# ==============================================================================
# To clean different files/folders:
#   1. Find the 'ITEMS_TO_DELETE' array above
#   2. Add your paths following the same format
#   3. Use $HOME instead of /home/username for portability
#
# Example:
#   "$HOME/Downloads/temp_files"
#   "/var/log/myapp/*.log"
# ==============================================================================