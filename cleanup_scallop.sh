#!/bin/bash

# Scallop Integration Cleanup Script
# Removes unnecessary files and consolidates to working implementation

echo "🧹 Cleaning up Scallop integration..."
echo "=================================="

cd /home/mpz/development/playground/uav_landing_project

# Step 1: Remove old/unused Scallop implementations
echo "1. Removing old Scallop implementations..."
rm -f src/scallop_reasoning_engine.py
rm -f src/scallop_reasoning_engine.py.backup  
rm -f src/scallop_mock.py
rm -f install_scallop.sh
echo "   ✅ Removed old Scallop files"

# Step 2: Remove old test files that reference removed implementations
echo "2. Removing outdated test files..."
rm -f test_reasoning_engine.py
rm -f tests/test_enhanced_scallop_system.py
echo "   ✅ Removed old test files"

# Step 3: Remove unnecessary documentation 
echo "3. Removing outdated documentation..."
rm -f docs/SCALLOP_NEUROSYMBOLIC_INTEGRATION_PLAN.md
rm -f TENSORRT_STATUS.md
echo "   ✅ Removed outdated docs"

# Step 4: Remove temporary/demo files that may reference old implementations
echo "4. Checking and cleaning demo files..."
if grep -q "scallop_reasoning_engine" demo_complete_system.py; then
    echo "   ⚠️  demo_complete_system.py may need manual updates"
fi

# Step 5: Verify what's left
echo ""
echo "📊 Remaining Scallop files:"
find . -name "*scallop*" -type f | grep -v .git | grep -v __pycache__ | sort

echo ""
echo "🎯 Cleanup Summary:"
echo "   ✅ Kept: scallop_reasoning_engine_simple.py (working implementation)"
echo "   ❌ Removed: Old complex implementation" 
echo "   ❌ Removed: Mock implementation"
echo "   ❌ Removed: Outdated tests and docs"

echo ""
echo "✅ Scallop cleanup complete!"
