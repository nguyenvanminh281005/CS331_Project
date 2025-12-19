#!/usr/bin/env python
"""Quick test for MORPH-2 processor"""

from utils.morph2_processor import MORPH2Processor

print("Testing MORPH-2 Processor...")

processor = MORPH2Processor()

# Load train metadata
print("\n1. Loading train metadata...")
train_df = processor.load_metadata('train')
print(f"   Train shape: {train_df.shape}")
print(f"   Columns: {train_df.columns.tolist()}")
print("\n   Sample data:")
print(train_df.head())

# Load validation metadata
print("\n2. Loading validation metadata...")
val_df = processor.load_metadata('val')
print(f"   Val shape: {val_df.shape}")

print("\n✅ MORPH-2 Processor test complete!")
