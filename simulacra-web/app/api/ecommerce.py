from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ..auth.auth import get_current_active_user, User

# Router for e-commerce endpoints
router = APIRouter()

# Models
class Product(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None
    image_url: Optional[str] = None

class CartItem(BaseModel):
    product_id: int
    quantity: int
    price: float

class Order(BaseModel):
    id: int
    user_id: str
    items: List[CartItem]
    total: float
    status: str
    created_at: str

class Address(BaseModel):
    id: int
    user_id: str
    street: str
    city: str
    state: str
    zip_code: str
    is_default: bool

class Review(BaseModel):
    id: int
    product_id: int
    user_id: str
    rating: int
    comment: Optional[str] = None

class TrackingDetails(BaseModel):
    order_id: int
    tracking_number: str
    carrier: str
    status: str
    estimated_delivery: Optional[str] = None
    updates: List[Dict[str, Any]]

# Mock data stores
mock_products = [
    {
        "id": 1,
        "name": "Sample Product 1",
        "price": 29.99,
        "description": "This is a sample product description",
        "image_url": "/static/img/product1.jpg"
    },
    {
        "id": 2,
        "name": "Sample Product 2",
        "price": 49.99,
        "description": "Another sample product description",
        "image_url": "/static/img/product2.jpg"
    }
]

mock_carts = {}  # user_id -> List[CartItem]
mock_orders = {}  # user_id -> List[Order]
mock_addresses = {}  # user_id -> List[Address]
mock_reviews = {}  # product_id -> List[Review]
mock_tracking = {}  # order_id -> TrackingDetails

# Cart endpoints
@router.post("/cart/clear")
async def clear_cart(current_user: User = Depends(get_current_active_user)):
    """Clear the user's shopping cart"""
    user_id = current_user.username
    if user_id in mock_carts:
        mock_carts[user_id] = []
    return {"status": "success", "message": "Cart cleared successfully"}

# Order endpoints
@router.get("/orders/list")
async def get_orders_list_product(current_user: User = Depends(get_current_active_user)):
    """Get list of user's orders with product details"""
    user_id = current_user.username
    if user_id not in mock_orders:
        mock_orders[user_id] = []
    return {"status": "success", "orders": mock_orders.get(user_id, [])}

@router.get("/orders/{order_id}")
async def get_order_detail(order_id: int, current_user: User = Depends(get_current_active_user)):
    """Get detailed information about a specific order"""
    user_id = current_user.username
    
    # Find the order in the user's orders
    orders = mock_orders.get(user_id, [])
    order = next((o for o in orders if o["id"] == order_id), None)
    
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return {"status": "success", "order": order}

# Tracking endpoints
@router.get("/orders/{order_id}/tracking")
async def get_track_details(order_id: int, current_user: User = Depends(get_current_active_user)):
    """Get tracking details for an order"""
    if order_id not in mock_tracking:
        # Create mock tracking data
        mock_tracking[order_id] = {
            "order_id": order_id,
            "tracking_number": f"TRK{order_id}12345",
            "carrier": "Sample Carrier",
            "status": "In Transit",
            "estimated_delivery": "2025-03-20",
            "updates": [
                {
                    "timestamp": "2025-03-13T10:00:00",
                    "status": "Package processed",
                    "location": "Distribution Center"
                },
                {
                    "timestamp": "2025-03-14T08:30:00",
                    "status": "In Transit",
                    "location": "En route to delivery"
                }
            ]
        }
    
    return {"status": "success", "tracking": mock_tracking.get(order_id)}

# Review endpoints
@router.get("/products/{product_id}/reviews")
async def get_review(product_id: int, current_user: User = Depends(get_current_active_user)):
    """Get reviews for a product"""
    if product_id not in mock_reviews:
        mock_reviews[product_id] = []
    
    return {"status": "success", "reviews": mock_reviews.get(product_id, [])}

# Address endpoints
@router.post("/address/clear")
async def clear_address(current_user: User = Depends(get_current_active_user)):
    """Clear all addresses for the user"""
    user_id = current_user.username
    if user_id in mock_addresses:
        mock_addresses[user_id] = []
    
    return {"status": "success", "message": "Addresses cleared successfully"}

@router.get("/address/count")
async def count_address(current_user: User = Depends(get_current_active_user)):
    """Count addresses for the user"""
    user_id = current_user.username
    count = len(mock_addresses.get(user_id, []))
    
    return {"status": "success", "count": count}
