#!/usr/bin/env python3
"""
================================================================================
AUDIO SIMILARITY DETECTION SYSTEM (2026 SOTA)
================================================================================

PURPOSE:
--------
This script analyzes audio files using state-of-the-art machine learning models
to find similar content. It uses CLAP (Contrastive Language-Audio Pretraining)
embeddings and GPU acceleration to process large audio datasets efficiently.

CONCEPT EXPLANATION (ELI15):
----------------------------
Imagine you have hundreds of songs or audio recordings. This script:
1. Converts each audio file into a "fingerprint" (called an embedding)
2. Compares these fingerprints to find which audios sound similar
3. Organizes similar files into folders automatically

Think of embeddings like a summary of what makes each audio unique -
the rhythm, the tones, the patterns. Similar audios have similar embeddings!

Author: AI Audio Engineer
Date: February 2026
Python Version: 3.8+
================================================================================
"""

# ============================================================================
# IMPORTS - Loading External Libraries
# ============================================================================
# CONCEPT: Python programs use "import" to load code from other libraries
# Think of imports like borrowing tools from a toolbox - we don't build
# everything from scratch, we use tested tools made by experts!
# ============================================================================

# SYSTEM LIBRARIES
# These come pre-installed with Python and handle basic operations
import argparse  # Command Line Argument Parser - lets users configure the script
import sys  # System-specific parameters and functions
from pathlib import Path  # Object-oriented file system paths (modern way!)
from typing import Dict, List, Optional, Tuple  # Type hints for code clarity

# EXPLANATION:
# - AutoModel: Automatically loads the correct model architecture
# - AutoProcessor: Automatically loads the correct data preprocessor
# These "Auto" classes are smart - they detect what model you're using and
# configure themselves accordingly. No manual setup needed!
# AUDIO PROCESSING
import librosa  # Advanced audio loading library

# NUMERICAL & SCIENTIFIC COMPUTING
# CONCEPT: These libraries handle math, arrays, and scientific calculations
import numpy as np  # NumPy: "Numerical Python" - the foundation for data science

# SYNTAX: "as np" creates a shorthand so we write "np.array" instead of "numpy.array"
import torch  # PyTorch: Deep Learning framework with GPU support
import torchaudio  # PyTorch's audio processing library

# CONCEPT: tqdm shows progress bars like: [████████░░] 80%
# Without it, long operations would just show nothing - frustrating!
from scipy.spatial.distance import cosine  # Cosine similarity calculation

# WHY BOTH librosa AND torchaudio?
# - librosa: Better at handling corrupted/unusual audio formats
# - torchaudio: Faster GPU-accelerated processing
# We use librosa to load safely, then convert to torchaudio tensors
# UTILITIES
from tqdm import tqdm  # Progress bar visualization

# CONCEPT: torchaudio is like "audio + torch" - it loads audio files as PyTorch tensors
# MACHINE LEARNING MODELS
from transformers import AutoModel, AutoProcessor

# CONCEPT: Cosine similarity measures how "aligned" two vectors are
# Range: -1 (opposite) to 1 (identical), we typically use 0.0 to 1.0


# ============================================================================
# CLASS DEFINITION: AudioSimilarityDetector
# ============================================================================
# CONCEPT: A "class" is like a blueprint for creating objects
# It bundles related data (attributes) and functions (methods) together
# Think of it like a recipe book - the class is the book, and each instance
# is a meal you make following those recipes
# ============================================================================


class AudioSimilarityDetector:
    """
    A production-grade system for detecting similar audio files using CLAP embeddings.

    ARCHITECTURE EXPLANATION:
    -------------------------
    This class follows the "Object-Oriented Programming" (OOP) paradigm:
    - __init__: Sets up the detector (like preparing your tools before cooking)
    - Methods: Different operations the detector can perform
    - Attributes: Data the detector remembers (stored in self.variable_name)

    ATTRIBUTES:
    -----------
    device : torch.device
        CONCEPT: Specifies where computations happen (CPU or GPU)
        GPU is like a specialized calculator - much faster for math-heavy tasks!

    model : transformers.AutoModel
        The pre-trained CLAP neural network that generates audio embeddings
        ANALOGY: Like a trained expert who can "understand" audio content

    processor : transformers.AutoProcessor
        Prepares audio data in the format the model expects
        ANALOGY: Like a translator who converts raw audio into "model language"

    sample_rate : int
        CONCEPT: How many audio samples per second (Hz = Hertz)
        48000 Hz means 48,000 measurements of sound wave height per second
        Higher = better quality but larger files
    """

    # SYNTAX NOTE: The ":" after the class name and function names indicates
    # the start of an indented code block. Python uses indentation (spaces/tabs)
    # instead of curly braces {} like other languages!

    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: Optional[str] = None):
        """
        Initialize the Audio Similarity Detector.

        CONCEPT: __init__ is a "constructor" - it runs automatically when you
        create a new AudioSimilarityDetector object. It sets up everything
        the detector needs to work.

        PARAMETERS EXPLAINED:
        ---------------------
        self : AudioSimilarityDetector
            CONCEPT: "self" refers to the object itself
            It's like saying "my" or "this detector's"
            Every method in a class needs self as the first parameter!

        model_name : str (string)
            SYNTAX: The "= '...'" part is a "default value"
            If the user doesn't specify a model, we use this one
            EXAMPLE: detector = AudioSimilarityDetector()  # Uses default
                     detector = AudioSimilarityDetector("other/model")  # Uses custom

        device : Optional[str]
            SYNTAX: "Optional[str]" means "either a string or None"
            None is Python's way of saying "nothing" or "not set"

        SYNTAX NOTE: "->" doesn't appear here because __init__ doesn't return anything
        """

        # ====================================================================
        # STEP 1: GPU SETUP & DEVICE SELECTION
        # ====================================================================
        print("🚀 Initializing Audio Similarity Detector...")
        print("=" * 70)

        # LOGIC: Check if user specified a device, otherwise auto-detect
        if device is None:
            # torch.cuda.is_available() returns True if NVIDIA GPU is available
            # SYNTAX: "if condition: A else: B" is a "ternary operator"
            # It's a compact if-else statement in one line
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # torch.device() converts a string like "cuda" into a device object
            self.device = torch.device(device)

        # VERIFICATION: Check if we're actually using GPU
        if self.device.type == "cuda":
            print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            # SYNTAX EXPLAINED:
            # - f"..." is an "f-string" (formatted string literal)
            # - {variable} inside f-strings gets replaced with the variable's value
            # - :.2f means "format as a number with 2 decimal places"
            # - / 1e9 converts bytes to gigabytes (1e9 = 1,000,000,000)
        else:
            print("⚠️  WARNING: No GPU detected! Processing will be slow.")
            print("   Install CUDA and PyTorch with GPU support for best performance.")
            # EDUCATIONAL NOTE: CPU processing can be 10-100x slower than GPU
            # for deep learning tasks. Always use GPU in production!

        # ====================================================================
        # STEP 2: LOAD THE CLAP MODEL & PROCESSOR
        # ====================================================================
        print(f"\n📦 Loading CLAP model: {model_name}")

        try:
            # TRY-EXCEPT BLOCK EXPLAINED:
            # "try:" attempts code that might fail
            # "except:" catches errors and handles them gracefully
            # WHY? Models might not download correctly, internet might fail, etc.

            # LOAD THE PROCESSOR FIRST
            # The processor converts raw audio into the format CLAP expects
            self.processor = AutoProcessor.from_pretrained(model_name)
            # CONCEPT: from_pretrained() downloads the model from HuggingFace Hub
            # It's cached locally after first download (stored in ~/.cache/)

            # LOAD THE MODEL
            # SYNTAX: dtype=torch.float16 uses "half precision" (16-bit floats)
            # WHY? Saves memory and speeds up computation with minimal accuracy loss
            # ANALOGY: Like using "HD" video instead of "4K" - still great quality!
            # MODELİ YÜKLE (GÜNCELLENMİŞ VERSİYON)
            self.model = AutoModel.from_pretrained(
                model_name,
                # dtype=torch.float16,  # <-- torch_dtype yerine sadece dtype
                # Üstteki satır hata verirse alttakini kullan, transformers sürümüne göre değişir.
                dtype=torch.float16,  # Eski parametre hala çoğu sürümde geçerli
                use_safetensors=True,  # <-- SİHİRLİ KOMUT BU!
            ).to(self.device)
            # SYNTAX: .to(device) transfers the model to the specified device
            # Think of it like moving furniture to a different room

            # SET TO EVALUATION MODE
            self.model.eval()
            # CONCEPT: Neural networks have two modes:
            # - Training mode: Updates weights, uses dropout, batch normalization
            # - Evaluation mode: Freezes weights, disables dropout
            # We're only using the model (not training), so we use eval() mode

            print("✅ Model loaded successfully!")

        except Exception as e:
            # SYNTAX: "Exception as e" catches any error and stores it in variable "e"
            # Exception is the base class for all errors in Python
            print("❌ ERROR: Failed to load model!")
            print(f"   Details: {e}")
            # PRODUCTION TIP: In real applications, we'd log this to a file
            # and potentially send alerts to system administrators
            sys.exit(1)  # Exit with error code 1 (0 = success, non-zero = failure)

        # ====================================================================
        # STEP 3: SET AUDIO PARAMETERS
        # ====================================================================
        # CLAP was trained on 48kHz audio, so we must resample to match
        self.sample_rate = 48000  # 48,000 samples per second

        # SUPPORTED AUDIO FORMATS
        # SYNTAX: This is a "tuple" - an immutable (unchangeable) list
        # Tuples use (), lists use []
        self.supported_formats = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Supported Formats: {', '.join(self.supported_formats)}")
        # SYNTAX: ', '.join(list) creates a string by joining list items with ", "
        # EXAMPLE: ['a', 'b', 'c'] becomes "a, b, c"

        print("=" * 70)
        print()

    # ========================================================================
    # METHOD: _load_audio (PRIVATE HELPER METHOD)
    # ========================================================================
    # SYNTAX: Methods starting with _ are "private" by convention
    # They're internal helpers not meant to be called from outside the class
    # ========================================================================

    def _load_audio(self, audio_path: Path) -> Optional[torch.Tensor]:
        try:
            # STEP 1: LOAD AUDIO WITH LIBROSA
            audio_array, original_sr = librosa.load(str(audio_path), sr=None, mono=True)

            # STEP 2: CONVERT TO TORCH TENSOR
            # Boyut: [örnek_sayısı] - Şimdilik düz (1D) bırakıyoruz
            audio_tensor = torch.FloatTensor(audio_array)

            # STEP 3: RESAMPLE IF NECESSARY
            if original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
                # Resample için [1, samples] formatı gerekir
                audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)

            # STEP 4: NORMALIZE
            max_val = torch.abs(audio_tensor).max()
            if max_val > 0:
                audio_tensor = audio_tensor / max_val

            return audio_tensor

        except Exception as e:
            print(f"⚠️  Warning: Could not load {audio_path.name}: {e}")
            return None

    def get_audio_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        # SES DOSYASINI YÜKLE
        audio_tensor = self._load_audio(audio_path)

        if audio_tensor is None:
            return None

        try:
            # İŞLEMCİYE GÖNDER (RAM Sorunu için .tolist() kullanıyoruz)
            inputs = self.processor(audio=audio_tensor.tolist(), sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Determine target dtype safely
                target_dtype = None
                model_dtype = getattr(self.model, "dtype", None)
                if model_dtype is not None:
                    target_dtype = model_dtype
                elif hasattr(self.model, "parameters"):
                    try:
                        target_dtype = next(self.model.parameters()).dtype
                    except (StopIteration, RuntimeError):
                        pass

                # Only cast floating-point tensors to target dtype, skip masks and integer tensors
                mask_keys = {"attention_mask"}
                if target_dtype is not None:
                    inputs = {k: v.to(target_dtype) if (k not in mask_keys and isinstance(v, torch.Tensor) and torch.is_floating_point(v)) else v for k, v in inputs.items()}

                # MODEL ÇIKTISINI AL
                outputs = self.model.get_audio_features(**inputs)

                # --- SAFEGUARD: TENSOR MÜ, NESNE Mİ? ---
                # Hatanın kaynağı burası: outputs bazen nesne olarak dönüyor.
                if isinstance(outputs, torch.Tensor):
                    embedding_tensor = outputs
                elif hasattr(outputs, "audio_embeds"):
                    # Bazı versiyonlarda 'audio_embeds' içinde saklanır
                    embedding_tensor = outputs.audio_embeds
                elif hasattr(outputs, "pooler_output"):
                    # Bazı durumlarda 'pooler_output' içindedir
                    embedding_tensor = outputs.pooler_output
                else:
                    # Hiçbiri değilse, ilk elemanı almayı deneriz (Tuple durumu)
                    embedding_tensor = outputs[0]

                # TENSÖRÜ NUMPY DİZİSİNE ÇEVİR
                # .float() ile 32-bit'e çekip numpy'a veriyoruz
                embedding = embedding_tensor.float().cpu().numpy().squeeze()

            return embedding

        except Exception as e:
            print(f"⚠️  Error processing {audio_path.name}: {e}")
            return None

    # ========================================================================
    # METHOD: compute_similarity_matrix (PUBLIC METHOD)
    # ========================================================================

    def compute_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise cosine similarity between all embeddings.

        CONCEPT: Cosine Similarity Explained
        -------------------------------------
        Cosine similarity measures the angle between two vectors:
        - 1.0 = Identical direction (very similar)
        - 0.0 = Perpendicular (unrelated)
        - -1.0 = Opposite direction (very different)

        MATHEMATICAL FORMULA:
        similarity = (A · B) / (||A|| * ||B||)
        Where:
        - A · B is the dot product (multiply corresponding elements, sum)
        - ||A|| is the magnitude (length) of vector A

        VISUAL ANALOGY:
        Imagine vectors as arrows in space. Cosine similarity measures
        how much they point in the same direction, regardless of length!

        PARAMETERS:
        -----------
        embeddings : List[np.ndarray]
            SYNTAX: List[type] means "a list where each element is of type 'type'"
            Here: A list of NumPy arrays (one per audio file)
            EXAMPLE: [embedding1, embedding2, embedding3]

        RETURNS:
        --------
        np.ndarray
            A 2D similarity matrix of shape (N, N) where N = number of audio files
            Element [i, j] = similarity between audio i and audio j

        MATRIX PROPERTIES:
        ------------------
        - Symmetric: similarity[i, j] == similarity[j, i]
        - Diagonal is always 1.0 (each audio is identical to itself!)
        """
        # STEP 1: DETERMINE THE NUMBER OF AUDIO FILES
        n = len(embeddings)

        # STEP 2: CREATE AN EMPTY SIMILARITY MATRIX
        # CONCEPT: np.zeros() creates an array filled with zeros
        # SHAPE: (n, n) creates a square matrix
        similarity_matrix = np.zeros((n, n))
        # EXAMPLE: If n=3, creates: [[0, 0, 0],
        #                              [0, 0, 0],
        #                              [0, 0, 0]]

        # STEP 3: COMPUTE ALL PAIRWISE SIMILARITIES
        # CONCEPT: We use nested loops to compare every pair
        # WHY? We need to compare audio 0 with 1,2,3..., audio 1 with 2,3,4..., etc.

        print("🔄 Computing similarity matrix...")

        # OUTER LOOP: Iterate through all audios
        # SYNTAX: range(n) generates numbers from 0 to n-1
        # EXAMPLE: range(3) generates 0, 1, 2
        for i in tqdm(range(n), desc="Computing similarities"):
            # tqdm wraps the loop and shows a progress bar
            # desc="..." sets the progress bar label

            # INNER LOOP: Compare current audio with all others
            # OPTIMIZATION: Start from i (not 0) because:
            # 1. similarity[i, i] = 1.0 (identity)
            # 2. similarity[i, j] = similarity[j, i] (symmetry)
            # This cuts computation roughly in half!
            for j in range(i, n):

                if i == j:
                    # DIAGONAL ELEMENTS: Audio compared with itself
                    similarity_matrix[i, j] = 1.0
                else:
                    # COMPUTE COSINE SIMILARITY
                    # 1 - cosine() because scipy's cosine() returns distance (0=similar)
                    # We want similarity (1=similar), so we subtract from 1
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    # MATHEMATICAL INSIGHT:
                    # cosine_distance = 1 - cosine_similarity
                    # Therefore: cosine_similarity = 1 - cosine_distance

                    # FILL BOTH (i,j) AND (j,i) due to symmetry
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Mirror value

        return similarity_matrix

    # ========================================================================
    # METHOD: find_similar_pairs (PUBLIC METHOD)
    # ========================================================================

    def find_similar_pairs(self, similarity_matrix: np.ndarray, audio_files: List[Path], threshold: float = 0.90) -> List[Tuple[Path, Path, float]]:
        """
        Identify pairs of audio files that exceed the similarity threshold.

        CONCEPT: Filtering by Threshold
        --------------------------------
        A threshold is a cutoff point. We only consider pairs with
        similarity >= threshold as "similar enough to group together"

        EXAMPLE: If threshold = 0.90 (90%), we only group files that are
        at least 90% similar. Lower threshold = more groups, looser matching.

        PARAMETERS:
        -----------
        similarity_matrix : np.ndarray
            The matrix computed by compute_similarity_matrix()

        audio_files : List[Path]
            List of audio file paths (in the same order as embeddings)

        threshold : float = 0.90
            SYNTAX: "= 0.90" provides a default value
            Minimum similarity score (0.0 to 1.0) to consider files similar

        RETURNS:
        --------
        List[Tuple[Path, Path, float]]
            SYNTAX BREAKDOWN:
            - List[...] = A list of items
            - Tuple[A, B, C] = A tuple with 3 elements of types A, B, C
            - Path = File path objects
            - float = Similarity score

            Each tuple contains: (file1_path, file2_path, similarity_score)
            EXAMPLE: [(audio1.wav, audio2.wav, 0.95), (audio1.wav, audio3.wav, 0.92)]
        """
        # STEP 1: INITIALIZE EMPTY LIST FOR RESULTS
        similar_pairs = []  # Empty list using []
        # CONCEPT: Lists are mutable (can be changed after creation)
        # We'll append tuples to this list as we find similar pairs

        # STEP 2: GET MATRIX DIMENSIONS
        n = len(audio_files)

        # STEP 3: SCAN THE UPPER TRIANGLE OF THE MATRIX
        # CONCEPT: We only check the upper triangle because:
        # 1. The diagonal is always 1.0 (we skip it)
        # 2. The matrix is symmetric (no need to check both sides)
        print(f"\n🔍 Finding pairs with similarity >= {threshold:.0%}...")
        # SYNTAX: {threshold:.0%} formats as percentage with 0 decimal places
        # EXAMPLE: 0.90 displays as "90%"

        # NESTED LOOP TO SCAN UPPER TRIANGLE
        for i in range(n):
            # OPTIMIZATION: Start j from i+1 to skip diagonal and lower triangle
            # This ensures we only look at each unique pair once
            for j in range(i + 1, n):

                # GET SIMILARITY VALUE FROM MATRIX
                similarity = similarity_matrix[i, j]

                # CHECK IF IT MEETS THRESHOLD
                if similarity >= threshold:
                    # CREATE A TUPLE WITH FILE PATHS AND SIMILARITY
                    # SYNTAX: (a, b, c) creates a tuple with 3 elements
                    pair = (audio_files[i], audio_files[j], similarity)
                    similar_pairs.append(pair)  # Add to results list
                    # APPEND vs EXTEND:
                    # - append(x): Adds x as a single element
                    # - extend(list): Adds all elements from list

        # STEP 4: REPORT FINDINGS
        print(f"✅ Found {len(similar_pairs)} similar pairs")

        return similar_pairs

    # ========================================================================
    # METHOD: organize_files (PUBLIC METHOD)
    # ========================================================================

    def organize_files(self, similar_pairs: List[Tuple[Path, Path, float]], output_dir: Path, copy_mode: bool = True) -> None:
        """
        Organize similar audio files into folders.

        CONCEPT: Clustering Similar Files
        ----------------------------------
        We group similar files together using a simple clustering approach:
        1. Start with the first pair -> create "cluster_1" folder
        2. For each new pair, check if either file is already in a cluster
        3. If yes, add the other file to that cluster
        4. If no, create a new cluster

        This is called "transitive grouping" - if A=B and B=C, then A,B,C
        are in the same group.

        PARAMETERS:
        -----------
        similar_pairs : List[Tuple[Path, Path, float]]
            The pairs found by find_similar_pairs()

        output_dir : Path
            Destination directory where cluster folders will be created

        copy_mode : bool = True
            CONCEPT: Determines file operation
            - True: Copy files (original files remain untouched - SAFE)
            - False: Move files (original files are deleted - DESTRUCTIVE)

        RETURNS:
        --------
        None
            SYNTAX: "-> None" means this function doesn't return a value
            It performs an action (organizing files) but doesn't give back data
            This is called a "side effect" - it changes the file system
        """
        # STEP 1: CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST
        # CONCEPT: mkdir() creates a directory
        # parents=True creates parent directories if needed (like "mkdir -p" in Linux)
        # exist_ok=True doesn't error if directory already exists
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Organizing files into: {output_dir}")
        print(f"   Mode: {'COPY' if copy_mode else 'MOVE'}")

        # STEP 2: CREATE FILE-TO-CLUSTER MAPPING
        # CONCEPT: A dictionary (dict) stores key-value pairs
        # SYNTAX: {} creates an empty dictionary
        # We'll map: audio_file_path -> cluster_number
        file_to_cluster: Dict[Path, int] = {}
        # TYPE HINT: Dict[Path, int] means "keys are Paths, values are ints"

        cluster_id = 0  # Start with cluster 0

        # STEP 3: ASSIGN FILES TO CLUSTERS
        print("\n🗂️  Building clusters...")

        # ITERATE THROUGH ALL SIMILAR PAIRS
        # SYNTAX: "for a, b, sim in list" unpacks each tuple into 3 variables
        for file1, file2, similarity in tqdm(similar_pairs, desc="Clustering"):

            # CHECK IF EITHER FILE IS ALREADY IN A CLUSTER
            cluster1 = file_to_cluster.get(file1)  # Returns None if not found
            cluster2 = file_to_cluster.get(file2)
            # SYNTAX: dict.get(key) is safer than dict[key]
            # If key doesn't exist: get() returns None, [key] raises an error

            # CASE 1: BOTH FILES ARE NEW (not in any cluster)
            if cluster1 is None and cluster2 is None:
                # Create a new cluster for both files
                file_to_cluster[file1] = cluster_id
                file_to_cluster[file2] = cluster_id
                cluster_id += 1  # Increment for next cluster
                # SYNTAX: += is shorthand for "variable = variable + 1"

            # CASE 2: FILE1 IS IN A CLUSTER, FILE2 IS NOT
            elif cluster1 is not None and cluster2 is None:
                # Add file2 to file1's cluster
                file_to_cluster[file2] = cluster1

            # CASE 3: FILE2 IS IN A CLUSTER, FILE1 IS NOT
            elif cluster2 is not None and cluster1 is None:
                # Add file1 to file2's cluster
                file_to_cluster[file1] = cluster2

            # CASE 4: BOTH ARE ALREADY IN CLUSTERS
            elif cluster1 != cluster2:
                # MERGE CLUSTERS: This is more complex!
                # All files in cluster2 should move to cluster1
                # CONCEPT: We iterate through all mappings and update them
                for file_path, cluster_num in list(file_to_cluster.items()):
                    # SYNTAX: dict.items() returns key-value pairs
                    # list(...) creates a copy so we can modify during iteration
                    if cluster_num == cluster2:
                        file_to_cluster[file_path] = cluster1

        # STEP 4: CREATE CLUSTER DIRECTORIES AND ORGANIZE FILES
        print(f"\n📦 Creating {cluster_id} cluster folders...")

        # GROUP FILES BY CLUSTER
        # CONCEPT: We invert the mapping: cluster_number -> [list of files]
        # WHY? So we can process all files in a cluster together
        clusters: Dict[int, List[Path]] = {}
        # SYNTAX: Dict[int, List[Path]] means keys are ints, values are lists of Paths

        for file_path, cluster_num in file_to_cluster.items():
            # CHECK IF CLUSTER EXISTS IN DICTIONARY
            if cluster_num not in clusters:
                # Create empty list for this cluster
                clusters[cluster_num] = []

            # ADD FILE TO CLUSTER'S LIST
            clusters[cluster_num].append(file_path)

        # PROCESS EACH CLUSTER
        # SYNTAX: dict.items() gives us both keys and values
        for cluster_num, files in tqdm(clusters.items(), desc="Organizing files"):

            # CREATE CLUSTER FOLDER
            # SYNTAX: f"string{variable}" embeds variable value in string
            cluster_folder = output_dir / f"cluster_{cluster_num}"
            # SYNTAX: "/" operator on Path objects joins paths safely
            # EXAMPLE: Path("/data") / "folder" = Path("/data/folder")
            cluster_folder.mkdir(exist_ok=True)

            # COPY OR MOVE EACH FILE IN THE CLUSTER
            for file_path in files:
                # DETERMINE DESTINATION PATH
                destination = cluster_folder / file_path.name
                # file_path.name gets just the filename (not full path)
                # EXAMPLE: Path("/home/audio/song.mp3").name = "song.mp3"

                try:
                    if copy_mode:
                        # COPY FILE (original remains)
                        import shutil  # Import only when needed (local import)

                        shutil.copy2(file_path, destination)
                        # copy2() preserves metadata (creation date, permissions)
                    else:
                        # MOVE FILE (original is deleted)
                        file_path.rename(destination)
                        # rename() is atomic (completes fully or not at all - safer!)

                except Exception as e:
                    print(f"⚠️  Warning: Could not process {file_path.name}")
                    print(f"   Reason: {e}")
                    # Continue with other files even if one fails

        print("\n✅ Organization complete!")
        print(f"   Total clusters created: {len(clusters)}")
        print(f"   Total files organized: {len(file_to_cluster)}")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
# CONCEPT: This function runs when the script is executed directly
# It handles command-line arguments and orchestrates the workflow
# ============================================================================


def main():
    """
    Main execution function - orchestrates the entire similarity detection workflow.

    WORKFLOW:
    ---------
    1. Parse command-line arguments
    2. Initialize the detector
    3. Scan for audio files
    4. Extract embeddings
    5. Compute similarity matrix
    6. Find similar pairs
    7. Organize files

    CONCEPT: Command-Line Interface (CLI)
    --------------------------------------
    This script is designed to be run from the terminal:
    $ python audio_similarity.py --source ./audio --threshold 0.85

    Users can customize behavior without editing code!
    """

    # ========================================================================
    # STEP 1: SETUP COMMAND-LINE ARGUMENT PARSER
    # ========================================================================
    # CONCEPT: argparse makes it easy to create professional CLI tools
    # It handles: parsing arguments, validation, help messages, error messages

    parser = argparse.ArgumentParser(
        description="🎵 Audio Similarity Detection System (CLAP + GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # formatter_class preserves formatting in description/epilog
        epilog="""
Examples:
  # Basic usage with defaults
  python audio_similarity.py --source ./my_audio

  # Custom threshold and destination
  python audio_similarity.py --source ./audio --target ./sorted --threshold 0.85

  # Move files instead of copying
  python audio_similarity.py --source ./audio --move
        """,
    )
    # SYNTAX: ArgumentParser() creates a parser object
    # We'll add arguments to this object below

    # ADD ARGUMENTS
    # SYNTAX: add_argument() defines a command-line option
    # "--name" creates an optional argument (vs positional like "filename")

    parser.add_argument(
        "--source",
        type=str,  # Expected data type
        required=True,  # User MUST provide this argument
        help="Path to source directory containing audio files",
        # help text is shown when user runs: python script.py --help
    )

    parser.add_argument("--target", type=str, default="./sorted_audio", help="Path to output directory (default: ./sorted_audio)")  # Used if user doesn't specify

    parser.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold (0.0-1.0, default: 0.90)")

    parser.add_argument(
        "--move",
        action="store_true",  # Boolean flag: present=True, absent=False
        help="Move files instead of copying (default: copy)",
        # CONCEPT: store_true means this is a flag, not a value
        # Usage: --move (turns on), omit (leaves off)
    )

    # PARSE ARGUMENTS
    # SYNTAX: parse_args() processes sys.argv (command-line arguments)
    args = parser.parse_args()
    # args is now an object with attributes: args.source, args.target, etc.

    # ========================================================================
    # STEP 2: VALIDATE ARGUMENTS
    # ========================================================================

    source_dir = Path(args.source)  # Convert string to Path object
    target_dir = Path(args.target)

    # CHECK IF SOURCE DIRECTORY EXISTS
    if not source_dir.exists():
        print(f"❌ ERROR: Source directory does not exist: {source_dir}")
        sys.exit(1)  # Exit with error

    if not source_dir.is_dir():
        print(f"❌ ERROR: Source path is not a directory: {source_dir}")
        sys.exit(1)

    # VALIDATE THRESHOLD
    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ ERROR: Threshold must be between 0.0 and 1.0 (got {args.threshold})")
        sys.exit(1)

    # ========================================================================
    # STEP 3: INITIALIZE THE DETECTOR
    # ========================================================================

    detector = AudioSimilarityDetector()
    # SYNTAX: ClassName() calls __init__ and creates an instance

    # ========================================================================
    # STEP 4: SCAN FOR AUDIO FILES
    # ========================================================================

    print("🔍 Scanning for audio files...")

    # FIND ALL AUDIO FILES RECURSIVELY
    # CONCEPT: We use Path.rglob() to search all subdirectories
    audio_files = []

    # ITERATE THROUGH SUPPORTED FORMATS
    for ext in detector.supported_formats:
        # SYNTAX: "**/*" means "all subdirectories, any name"
        # "*" is a wildcard matching any characters
        # rglob = recursive glob (search in all subdirectories)
        files = list(source_dir.rglob(f"*{ext}"))
        audio_files.extend(files)
        # extend() adds all elements from files to audio_files

    if not audio_files:
        print(f"❌ ERROR: No audio files found in {source_dir}")
        print(f"   Supported formats: {', '.join(detector.supported_formats)}")
        sys.exit(1)

    print(f"✅ Found {len(audio_files)} audio files")

    # ========================================================================
    # STEP 5: EXTRACT EMBEDDINGS
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 1: EXTRACTING EMBEDDINGS")
    print("=" * 70)

    embeddings = []  # List to store all embeddings
    valid_files = []  # List to store files that loaded successfully

    # PROCESS EACH FILE
    # SYNTAX: enumerate() gives us both index and value
    # EXAMPLE: enumerate(['a', 'b']) yields (0, 'a'), (1, 'b')
    for idx, file_path in enumerate(tqdm(audio_files, desc="Extracting embeddings")):
        # EXTRACT EMBEDDING
        embedding = detector.get_audio_embedding(file_path)

        # CHECK IF EXTRACTION SUCCEEDED
        if embedding is not None:
            embeddings.append(embedding)
            valid_files.append(file_path)
        else:
            # File failed to load (already logged by get_audio_embedding)
            pass  # Skip this file

    if len(valid_files) < 2:
        print("❌ ERROR: Need at least 2 valid audio files to compute similarity")
        sys.exit(1)

    print(f"\n✅ Successfully extracted {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {embeddings[0].shape[0]}")
    # EXAMPLE: "Embedding dimension: 512" means each embedding has 512 numbers

    # ========================================================================
    # STEP 6: COMPUTE SIMILARITY MATRIX
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 2: COMPUTING SIMILARITIES")
    print("=" * 70)

    similarity_matrix = detector.compute_similarity_matrix(embeddings)

    # ========================================================================
    # STEP 7: FIND SIMILAR PAIRS
    # ========================================================================

    similar_pairs = detector.find_similar_pairs(similarity_matrix, valid_files, threshold=args.threshold)

    if not similar_pairs:
        print(f"\n⚠️  No similar pairs found with threshold {args.threshold:.0%}")
        print("   Try lowering the threshold (e.g., --threshold 0.85)")
        return  # Exit gracefully (not an error, just no matches)

    # ========================================================================
    # STEP 8: ORGANIZE FILES
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 3: ORGANIZING FILES")
    print("=" * 70)

    detector.organize_files(similar_pairs, target_dir, copy_mode=not args.move)  # Invert: --move flag means copy_mode=False

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 70)
    print("📊 Statistics:")
    print(f"   Total files scanned: {len(audio_files)}")
    print(f"   Valid files processed: {len(valid_files)}")
    print(f"   Similar pairs found: {len(similar_pairs)}")
    print(f"   Output directory: {target_dir}")
    print("=" * 70)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
# CONCEPT: This is a Python idiom that checks if the script is being run
# directly (vs imported as a module)
#
# __name__ is a special variable:
# - When script is run directly: __name__ == "__main__"
# - When script is imported: __name__ == module name
#
# WHY? Allows us to write code that can be both:
# 1. Run as a standalone script
# 2. Imported into other programs
# ============================================================================

if __name__ == "__main__":
    # SYNTAX: This if statement only executes when script is run directly
    # EXAMPLE: python audio_similarity.py  <- This runs main()
    #          import audio_similarity     <- This doesn't run main()

    try:
        main()  # Call the main function
    except KeyboardInterrupt:
        # CONCEPT: KeyboardInterrupt is raised when user presses Ctrl+C
        # We catch it to exit gracefully instead of showing an ugly error
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)  # Exit successfully (0 = no error)
    except Exception as e:
        # CATCH-ALL: Any unexpected error
        print(f"\n❌ FATAL ERROR: {e}")
        # In production, we'd also:
        # - Log the full stack trace to a file
        # - Send alerts to monitoring systems
        # - Clean up any partial results
        import traceback

        traceback.print_exc()  # Print detailed error information
        sys.exit(1)
