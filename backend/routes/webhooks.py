#!/usr/bin/env python3
"""
SCIO - Webhook Handlers
Stripe, Vast.ai, RunPod Webhooks
"""

import os
import json
import hashlib
import uuid
from datetime import datetime

from flask import request, jsonify
import stripe

from . import webhooks_bp
from backend.config import Config
from backend.models import SessionLocal
from backend.models.job import Job, JobType, JobStatus
from backend.models.earnings import Earning, EarningSource, EarningType
from backend.services.job_queue import get_job_queue


# Configure Stripe
stripe.api_key = Config.STRIPE_SECRET_KEY


# ═══════════════════════════════════════════════════════════════
# STRIPE WEBHOOKS
# ═══════════════════════════════════════════════════════════════

@webhooks_bp.route('/stripe', methods=['POST'])
def stripe_webhook():
    """Stripe Webhook Handler"""
    payload = request.data
    sig = request.headers.get('Stripe-Signature')
    secret = Config.STRIPE_WEBHOOK_SECRET

    try:
        if secret:
            event = stripe.Webhook.construct_event(payload, sig, secret)
        else:
            print("[WARN]  WARNING: STRIPE_WEBHOOK_SECRET not set!")
            event = stripe.Event.construct_from(
                json.loads(payload),
                stripe.api_key
            )
    except ValueError:
        print("[ERROR] Invalid payload")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        print("[ERROR] Invalid signature")
        return jsonify({'error': 'Invalid signature'}), 400

    event_type = event['type']
    data = event['data']['object']

    print(f"[DOWNLOAD] Stripe Webhook: {event_type}")

    if event_type == 'payment_intent.succeeded':
        return handle_payment_succeeded(data)

    elif event_type == 'payment_intent.payment_failed':
        return handle_payment_failed(data)

    elif event_type == 'customer.subscription.created':
        return handle_subscription_created(data)

    elif event_type == 'customer.subscription.deleted':
        return handle_subscription_deleted(data)

    elif event_type == 'invoice.paid':
        return handle_invoice_paid(data)

    return jsonify({'status': 'ignored'})


def handle_payment_succeeded(payment_intent):
    """Verarbeitet erfolgreiche Zahlung"""
    metadata = payment_intent.get('metadata', {})
    order_id = metadata.get('order_id')
    email = metadata.get('email')
    model_size = metadata.get('model_size')
    amount = payment_intent['amount']
    service = metadata.get('service', 'SCIO')

    print(f"\n{'='*70}")
    print(f"[MONEY] ZAHLUNG ERFOLGREICH!")
    print(f"{'='*70}")
    print(f"Order-ID: {order_id}")
    print(f"Email: {email}")
    print(f"Model: {model_size}")
    print(f"Betrag: {amount/100:.2f}€")
    print(f"{'='*70}\n")

    # Record earning
    db = SessionLocal()
    try:
        earning = Earning(
            earning_id=f"earn_{uuid.uuid4().hex[:16]}",
            earning_type=EarningType.INCOME,
            source=EarningSource.WEBSITE,
            amount_cents=amount,
            currency='EUR',
            order_id=order_id,
            external_id=payment_intent['id'],
            description=f"Training: {model_size}",
            stripe_fee_cents=int(amount * 0.029 + 25),  # Stripe ~2.9% + 0.25€
            status='completed',
            completed_at=datetime.utcnow(),
        )
        earning.net_amount_cents = amount - earning.stripe_fee_cents

        db.add(earning)
        db.commit()

        print(f"[OK] Earning erfasst: {earning.earning_id}")

    except Exception as e:
        print(f"[ERROR] Earning erfassen fehlgeschlagen: {e}")
        db.rollback()
    finally:
        db.close()

    # Create training job
    if order_id and model_size:
        try:
            queue = get_job_queue()
            job_id = queue.create_job(
                job_type=JobType.LLM_TRAINING,
                input_data={
                    'model_id': model_size.replace('llama-', 'llama-'),
                    'dataset_path': str(Config.ORDERS_DIR / order_id / 'dataset.jsonl'),
                    'order_id': order_id,
                },
                user_email=email,
                order_id=order_id,
                priority=10,  # High priority for paid jobs
            )
            print(f"[OK] Training-Job erstellt: {job_id}")

        except Exception as e:
            print(f"[ERROR] Training-Job erstellen fehlgeschlagen: {e}")

    return jsonify({'status': 'success', 'order_id': order_id})


def handle_payment_failed(payment_intent):
    """Verarbeitet fehlgeschlagene Zahlung"""
    order_id = payment_intent.get('metadata', {}).get('order_id')
    print(f"[ERROR] Zahlung fehlgeschlagen: {order_id}")

    return jsonify({'status': 'noted'})


def handle_subscription_created(subscription):
    """Verarbeitet neue Subscription"""
    customer_id = subscription['customer']
    plan_id = subscription['items']['data'][0]['price']['id']

    print(f"[JOB] Neue Subscription: {customer_id} -> {plan_id}")

    return jsonify({'status': 'noted'})


def handle_subscription_deleted(subscription):
    """Verarbeitet gekündigte Subscription"""
    customer_id = subscription['customer']
    print(f"[JOB] Subscription gekündigt: {customer_id}")

    return jsonify({'status': 'noted'})


def handle_invoice_paid(invoice):
    """Verarbeitet bezahlte Rechnung"""
    amount = invoice['amount_paid']
    customer_id = invoice['customer']

    print(f"[STRIPE] Rechnung bezahlt: {amount/100:.2f}€ von {customer_id}")

    # Record recurring earning
    db = SessionLocal()
    try:
        earning = Earning(
            earning_id=f"earn_{uuid.uuid4().hex[:16]}",
            earning_type=EarningType.INCOME,
            source=EarningSource.WEBSITE,
            amount_cents=amount,
            currency='EUR',
            external_id=invoice['id'],
            description=f"Subscription Payment",
            stripe_fee_cents=int(amount * 0.029 + 25),
            status='completed',
            completed_at=datetime.utcnow(),
        )
        earning.net_amount_cents = amount - earning.stripe_fee_cents

        db.add(earning)
        db.commit()

    except Exception as e:
        print(f"[ERROR] Earning erfassen fehlgeschlagen: {e}")
        db.rollback()
    finally:
        db.close()

    return jsonify({'status': 'success'})


# ═══════════════════════════════════════════════════════════════
# VAST.AI WEBHOOKS
# ═══════════════════════════════════════════════════════════════

@webhooks_bp.route('/vastai', methods=['POST'])
def vastai_webhook():
    """Vast.ai Webhook Handler"""
    data = request.json

    event_type = data.get('type')
    print(f"[DOWNLOAD] Vast.ai Webhook: {event_type}")

    if event_type == 'rental_started':
        return handle_vastai_rental_started(data)

    elif event_type == 'rental_ended':
        return handle_vastai_rental_ended(data)

    return jsonify({'status': 'ignored'})


def handle_vastai_rental_started(data):
    """Verarbeitet gestartete Vast.ai Rental"""
    instance_id = data.get('instance_id')
    price_per_hour = data.get('price_per_hour', 0)

    print(f"🖥️  Vast.ai Rental gestartet: {instance_id} @ ${price_per_hour}/h")

    return jsonify({'status': 'noted'})


def handle_vastai_rental_ended(data):
    """Verarbeitet beendete Vast.ai Rental"""
    instance_id = data.get('instance_id')
    total_earned = data.get('total_earned', 0)
    duration_hours = data.get('duration_hours', 0)

    print(f"🖥️  Vast.ai Rental beendet: {instance_id}, ${total_earned:.2f} verdient")

    # Record earning
    if total_earned > 0:
        db = SessionLocal()
        try:
            # Convert USD to EUR (simplified)
            amount_eur_cents = int(total_earned * 0.92 * 100)

            earning = Earning(
                earning_id=f"earn_{uuid.uuid4().hex[:16]}",
                earning_type=EarningType.INCOME,
                source=EarningSource.VASTAI,
                amount_cents=amount_eur_cents,
                currency='EUR',
                external_id=str(instance_id),
                description=f"Vast.ai Rental ({duration_hours:.1f}h)",
                platform_fee_cents=int(amount_eur_cents * 0.15),  # ~15% Vast.ai fee
                status='completed',
                completed_at=datetime.utcnow(),
            )
            earning.net_amount_cents = amount_eur_cents - earning.platform_fee_cents

            db.add(earning)
            db.commit()

            print(f"[OK] Vast.ai Earning erfasst: {earning.net_amount_cents/100:.2f}€")

        except Exception as e:
            print(f"[ERROR] Earning erfassen fehlgeschlagen: {e}")
            db.rollback()
        finally:
            db.close()

    return jsonify({'status': 'success'})


# ═══════════════════════════════════════════════════════════════
# RUNPOD WEBHOOKS
# ═══════════════════════════════════════════════════════════════

@webhooks_bp.route('/runpod', methods=['POST'])
def runpod_webhook():
    """RunPod Webhook Handler"""
    data = request.json

    event_type = data.get('type')
    print(f"[DOWNLOAD] RunPod Webhook: {event_type}")

    # RunPod uses serverless, so mainly track endpoint calls
    if event_type == 'job_completed':
        job_id = data.get('id')
        execution_time = data.get('executionTime', 0)
        print(f"[OK] RunPod Job abgeschlossen: {job_id} ({execution_time}ms)")

    return jsonify({'status': 'noted'})


# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@webhooks_bp.route('/health', methods=['GET'])
def webhook_health():
    """Webhook Health Check"""
    return jsonify({
        'status': 'healthy',
        'endpoints': [
            '/webhooks/stripe',
            '/webhooks/vastai',
            '/webhooks/runpod',
        ],
        'timestamp': datetime.utcnow().isoformat(),
    })
