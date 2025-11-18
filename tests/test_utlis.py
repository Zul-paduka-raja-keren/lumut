import pytest
from lumut.utils import tambah


class TestTambah:
    """Test suite for tambah function"""
    
    def test_positive_numbers(self):
        """Test addition of positive numbers"""
        assert tambah(2, 3) == 5
        assert tambah(10, 20) == 30
        assert tambah(100, 200) == 300
    
    def test_negative_numbers(self):
        """Test addition of negative numbers"""
        assert tambah(-5, -3) == -8
        assert tambah(-10, -20) == -30
    
    def test_mixed_signs(self):
        """Test addition of positive and negative numbers"""
        assert tambah(10, -5) == 5
        assert tambah(-10, 5) == -5
        assert tambah(100, -100) == 0
    
    def test_zero_addition(self):
        """Test addition with zero"""
        assert tambah(0, 0) == 0
        assert tambah(5, 0) == 5
        assert tambah(0, 5) == 5
    
    def test_float_numbers(self):
        """Test addition of float numbers"""
        assert tambah(2.5, 3.5) == 6.0
        assert tambah(1.1, 2.2) == pytest.approx(3.3)
        assert tambah(0.1, 0.2) == pytest.approx(0.3)
    
    def test_large_numbers(self):
        """Test addition of large numbers"""
        assert tambah(1000000, 2000000) == 3000000
        assert tambah(1e10, 2e10) == 3e10
    
    def test_return_type(self):
        """Test that function returns float"""
        result = tambah(1, 2)
        assert isinstance(result, (int, float))


class TestTambahEdgeCases:
    """Test edge cases for tambah function"""
    
    def test_very_small_numbers(self):
        """