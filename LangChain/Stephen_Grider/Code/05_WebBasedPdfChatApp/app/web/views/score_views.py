from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest

from app.web.hooks import login_required, load_model
from app.web.db.models import Conversation
from app.chat import score_conversation, get_scores

bp = Blueprint("score", __name__, url_prefix="/api/scores")


@bp.route("/", methods=["POST"])
@login_required
@load_model(Conversation, lambda r: r.args.get("conversation_id"))
def update_score(conversation):
    score = request.json.get("score")
    if not isinstance(score, (int, float)) or score < -1 or score > 1:
        raise BadRequest("Score must be a float between -1 and 1")

    score_conversation(
        conversation.id,
        score,
        llm=conversation.llm,
        retriever=conversation.retriever,
        memory=conversation.memory,
    )

    return {"message": "Score updated"}


@bp.route("/", methods=["GET"])
@login_required
def list_scores():
    scores = get_scores()

    return jsonify(scores)
