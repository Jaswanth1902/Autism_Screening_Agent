
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from app import calculate_isaa_risk

def test_scoring():
    # Test all Rarely (Score 1)
    data_all_rarely = {f"Q{i}": "0" for i in range(1, 41)}
    res1 = calculate_isaa_risk(data_all_rarely)
    print(f"All Rarely: {res1['score']} (Expected: 40)")
    
    # Test all Always (Score 5)
    data_all_always = {f"Q{i}": "1" for i in range(1, 41)}
    res2 = calculate_isaa_risk(data_all_always)
    print(f"All Always: {res2['score']} (Expected: 200)")
    
    # Test sometimes (Score 2)
    data_sometimes = {f"Q{i}": "0.25" for i in range(1, 41)}
    res3 = calculate_isaa_risk(data_sometimes)
    print(f"All Sometimes (0.25): {res3['score']} (Expected: 80)")

    # Test frequently (Score 3)
    data_frequently = {f"Q{i}": "0.5" for i in range(1, 41)}
    res4 = calculate_isaa_risk(data_frequently)
    print(f"All Frequently (0.5): {res4['score']} (Expected: 120)")

    # Test mostly (Score 4)
    data_mostly = {f"Q{i}": "0.75" for i in range(1, 41)}
    res5 = calculate_isaa_risk(data_mostly)
    print(f"All Mostly (0.75): {res5['score']} (Expected: 160)")

if __name__ == "__main__":
    test_scoring()
