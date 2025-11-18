import pytest
from lumut.core import hello_lumut


class TestHelloLumut:
    """Test suite for hello_lumut function"""
    
    def test_basic_greeting(self):
        """Test basic greeting functionality"""
        result = hello_lumut("World")
        assert result == "Halo, World, ini dari lumut!"
    
    def test_greeting_with_indonesian_name(self):
        """Test greeting with Indonesian name"""
        result = hello_lumut("Budi")
        assert result == "Halo, Budi, ini dari lumut!"
    
    def test_greeting_with_empty_string(self):
        """Test greeting with empty string"""
        result = hello_lumut("")
        assert result == "Halo, , ini dari lumut!"
    
    def test_greeting_with_special_characters(self):
        """Test greeting with special characters"""
        result = hello_lumut("User@123")
        assert result == "Halo, User@123, ini dari lumut!"
    
    def test_return_type(self):
        """Test that function returns a string"""
        result = hello_lumut("Test")
        assert isinstance(result, str)
    
    def test_greeting_format(self):
        """Test that greeting follows expected format"""
        name = "TestUser"
        result = hello_lumut(name)
        assert result.startswith("Halo,")
        assert name in result
        assert result.endswith("ini dari lumut!")


class TestHelloLumutEdgeCases:
    """Test edge cases for hello_lumut function"""
    
    def test_very_long_name(self):
        """Test with very long name"""
        long_name = "A" * 1000
        result = hello_lumut(long_name)
        assert long_name in result
        assert len(result) > 1000
    
    def test_unicode_characters(self):
        """Test with unicode characters"""
        result = hello_lumut("用户")
        assert "用户" in result
    
    def test_numeric_string(self):
        """Test with numeric string"""
        result = hello_lumut("12345")
        assert result == "Halo, 12345, ini dari lumut!"
    
    def test_whitespace_name(self):
        """Test with whitespace in name"""
        result = hello_lumut("John Doe")
        assert result == "Halo, John Doe, ini dari lumut!"