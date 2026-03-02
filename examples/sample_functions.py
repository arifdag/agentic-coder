"""Sample Python functions for testing the LLM Agent Platform."""

from typing import List, Optional


def calculate_average(numbers: List[float]) -> float:
    """Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers to average
        
    Returns:
        The arithmetic mean
        
    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome.
    
    Args:
        s: String to check
        
    Returns:
        True if the string is a palindrome
    """
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def find_max(items: List[int], default: Optional[int] = None) -> Optional[int]:
    """Find the maximum value in a list.
    
    Args:
        items: List of integers
        default: Value to return if list is empty
        
    Returns:
        Maximum value or default
    """
    if not items:
        return default
    
    max_val = items[0]
    for item in items[1:]:
        if item > max_val:
            max_val = item
    return max_val


def merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]:
    """Merge two sorted lists into a single sorted list.
    
    Args:
        list1: First sorted list
        list2: Second sorted list
        
    Returns:
        Merged sorted list
    """
    result = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    result.extend(list1[i:])
    result.extend(list2[j:])
    
    return result
