import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore, exceptions
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
import hashlib
import secrets

# Initialize Firebase Admin SDK
cred = credentials.Certificate("mystreamlitdb-59df17d15b19.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Initialize Google OAuth2 client
client_id = "102420858709-8mtqobjd6ss9pk7cpar5h90ce5tc2n35.apps.googleusercontent.com"
client_secret = "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"
redirect_url = "https://financeapp-l4zi55htjiv9vxwdwvgihv.streamlit.app/" 

client = GoogleOAuth2(client_id=client_id, client_secret=client_secret)

# Initialize session state variables
if 'email' not in st.session_state:
    st.session_state.email = ''
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'useremail' not in st.session_state:
    st.session_state.useremail = ''
if 'signedout' not in st.session_state:
    st.session_state.signedout = False
if 'signout' not in st.session_state:
    st.session_state.signout = False

async def get_access_token(client: GoogleOAuth2, redirect_url: str, code: str):
    return await client.get_access_token(code, redirect_url)

async def get_email(client: GoogleOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email

def hash_password(password, salt=None):
    """Hash a password for storing."""
    if salt is None:
        salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), 
                                   salt.encode('utf-8'), 100000)
    return salt + ":" + pwdhash.hex()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password.split(':')[0]
    stored_hash = stored_password.split(':')[1]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), 
                                   salt.encode('utf-8'), 100000)
    return pwdhash.hex() == stored_hash

def get_logged_in_user_email():
    try:
        query_params = st.query_params
        code = query_params.get('code')
        if code:
            token = asyncio.run(get_access_token(client, redirect_url, code))
            st.query_params.clear()

            if token:
                user_id, user_email = asyncio.run(get_email(client, token['access_token']))
                if user_email:
                    # Check if user exists in Firestore
                    users_ref = db.collection('users')
                    query = users_ref.where('email', '==', user_email).limit(1)
                    results = query.get()
                    
                    user_exists = False
                    for doc in results:
                        user_exists = True
                        user_data = doc.to_dict()
                        st.session_state.username = user_data.get('username', '')
                        break
                    
                    if user_exists:
                        # User exists, just update session state
                        st.session_state.email = user_email
                        st.session_state.useremail = user_email
                        st.session_state.signout = True
                        st.session_state.signedout = True
                        return user_email
                    else:
                        # User doesn't exist, create a new user in the database
                        try:
                            # Check if user exists in Auth
                            try:
                                user = auth.get_user_by_email(user_email)
                            except exceptions.FirebaseError:
                                # Create user in Auth
                                user = auth.create_user(email=user_email)
                            
                            # Generate a username from email
                            suggested_username = user_email.split('@')[0]
                            
                            # Store in Firestore
                            db.collection('users').document(user.uid).set({
                                'email': user_email,
                                'username': suggested_username,
                                'created_with_google': True
                            })
                            
                            st.session_state.email = user_email
                            st.session_state.useremail = user_email
                            st.session_state.username = suggested_username
                            st.session_state.signout = True
                            st.session_state.signedout = True
                            
                            st.success(f"New account created with Google for {user_email}")
                            return user_email
                        except Exception as e:
                            st.error(f"Error creating user: {e}")
                            return None
        return None
    except Exception as e:
        st.error(f"Error during authentication: {e}")
        return None
    
def show_login_button():
    authorization_url = asyncio.run(client.get_authorization_url(
        redirect_url,
        scope=["email", "profile"],
        extras_params={"access_type": "offline"},
    ))
    st.markdown(f'<a href="{authorization_url}" target="_self">Sign up with Google</a>', unsafe_allow_html=True)
    get_logged_in_user_email()

def app():
    st.title('Welcome to Mahmoud! Your Finance AI assistant')

    try:
        if not st.session_state.email:
            get_logged_in_user_email()
            if not st.session_state.email:
                show_login_button()

        if st.session_state.email:
            st.write(f"Logged in as: {st.session_state.email}")
            if st.button("Logout"):
                st.session_state.email = ''
                st.session_state.username = ''
                st.session_state.useremail = ''
                st.session_state.signout = False
                st.session_state.signedout = False
                st.rerun()

        if not st.session_state['signedout']:
            choice = st.selectbox("Select an option", ["Login", "Sign Up"])
        else:
            choice = "Sign Out"

        if choice == "Login":
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                try:
                    # Check if user exists
                    user = auth.get_user_by_email(email)
                    
                    # Check password in Firestore (since Firebase Auth doesn't let us verify passwords)
                    users_ref = db.collection('users')
                    query = users_ref.where('email', '==', email).limit(1)
                    results = query.get()
                    
                    user_found = False
                    for doc in results:
                        user_data = doc.to_dict()
                        stored_password = user_data.get('password', '')
                        
                        # Check if user was created with Google (no password)
                        if user_data.get('created_with_google', False):
                            user_found = True
                            st.warning("This account was created with Google. Please use Google login.")
                            break
                            
                        # Check password if it exists
                        if stored_password and verify_password(stored_password, password):
                            user_found = True
                            st.success(f"Logged in as {email}")
                            st.session_state.username = user_data.get('username', '')
                            st.session_state.useremail = email
                            st.session_state.email = email
                            st.session_state.signout = True
                            st.session_state.signedout = True
                            break
                    
                    if not user_found:
                        st.warning("Incorrect password. Please try again.")
                        
                except Exception as e:
                    st.warning(f"Login failed: {e}")
        elif choice == "Sign Up":
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            username = st.text_input("Enter your unique username")
            if st.button("Create my account"):
                if email and password and username:  # Make sure all fields are filled
                    try:
                        # Check if email already exists
                        try:
                            existing_user = auth.get_user_by_email(email)
                            st.warning(f"An account with email {email} already exists. Please log in.")
                        except exceptions.FirebaseError:
                            # Email doesn't exist, create new user
                            user = auth.create_user(email=email, password=password)
                            
                            # Hash the password
                            hashed_password = hash_password(password)
                            
                            # Add user data to Firestore
                            db.collection('users').document(user.uid).set({
                                'email': email,
                                'username': username,
                                'password': hashed_password,
                                'created_with_google': False
                            })
                            st.success("User created successfully!")
                            st.markdown('Please login to continue')
                    except Exception as e:
                        st.warning(f"Error creating user: {e}")
                else:
                    st.warning("Please fill in all fields")
        elif choice == "Sign Out":
            st.text('Username: ' + st.session_state.username)
            st.text('Email: ' + st.session_state.useremail)
            if st.button('Sign Out'):
                st.session_state.signout = False
                st.session_state.signedout = False
                st.session_state.username = ''
                st.session_state.email = ''
                st.session_state.useremail = ''
                st.rerun()
    except AttributeError as e:
        st.error(f"Your session has timed out. Please log in again. Error: {e}")
        # Reset session state variables
        st.session_state.email = ''
        st.session_state.username = ''
        st.session_state.useremail = ''
        st.session_state.signout = False
        st.session_state.signedout = False
        st.rerun()