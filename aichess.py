import pygame


pygame.init()#---initializing the pygame setup---#
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQ_SIZE = WIDTH // DIMENSION
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (124, 252, 0, 150)

PIECE_VALUES = {
    'P': 10, 'N': 30, 'B': 30, 'R': 50, 'Q': 90, 'K': 900,
    'p': -10, 'n': -30, 'b': -30, 'r': -50, 'q': -90, 'k': -900
}


class Move:
    def __init__(self, start_sq, end_sq, board, is_en_passant=False, is_castle=False):
        self.start_row, self.start_col = start_sq
        self.end_row, self.end_col = end_sq
        self.piece_moved = board.board[self.start_row][self.start_col]
        self.piece_captured = board.board[self.end_row][self.end_col]
        self.is_pawn_promotion = (self.piece_moved == 'P' and self.end_row == 0) or \
                                 (self.piece_moved == 'p' and self.end_row == 7)
        self.is_en_passant = is_en_passant
        self.is_castle = is_castle

        #---for speed purposes---#
        self.move_id = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.move_id == other.move_id
        return False


class Board:
    def __init__(self):
    #---initializes the empty board and sets up the game state---#
        self.board = [['' for _ in range(8)] for _ in range(8)]
        self._init_board()
        self.white_to_move = True
        self.move_log = []
        self.white_king_pos = (7, 4)
        self.black_king_pos = (0, 4)
        self.checkmate = False
        self.stalemate = False
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_target = None

    def _init_board(self):

        for col in range(8):
            self.board[1][col] = 'p'  #---places the pawns in the respective rows---#
            self.board[6][col] = 'P'

        #---Black---#
        self.board[0] = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']

        #---White---#
        self.board[7] = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']

    def make_move(self, move):

        self.board[move.end_row][move.end_col] = move.piece_moved
        self.board[move.start_row][move.start_col] = ''
        self.move_log.append(move)

        #---Update king pos---#
        if move.piece_moved == 'K':
            self.white_king_pos = (move.end_row, move.end_col)
        elif move.piece_moved == 'k':
            self.black_king_pos = (move.end_row, move.end_col)

        #---Handles pawn promotion---#
        if move.is_pawn_promotion:
            self.board[move.end_row][move.end_col] = 'Q' if move.piece_moved.isupper() else 'q'

        #---Handles castling---#
        if move.is_castle:
            if move.end_col - move.start_col == 2:
                #---Move rook from H to F---#
                self.board[move.end_row][move.end_col - 1] = self.board[move.end_row][7]
                self.board[move.end_row][7] = ''
            else:  #---Queen-side castle---#
                #---Move rook from A to D---#
                self.board[move.end_row][move.end_col + 1] = self.board[move.end_row][0]
                self.board[move.end_row][0] = ''

        # Toggle turn
        self.white_to_move = not self.white_to_move

    def undo_move(self):
        if not self.move_log:
            return
        move = self.move_log.pop()
        self.board[move.start_row][move.start_col] = move.piece_moved
        self.board[move.end_row][move.end_col] = move.piece_captured

        #---Update king position---#
        if move.piece_moved == 'K':
            self.white_king_pos = (move.start_row, move.start_col)
        elif move.piece_moved == 'k':
            self.black_king_pos = (move.start_row, move.start_col)

        #---Handle castling---#
        if move.is_castle:
            if move.end_col - move.start_col == 2:  #---King-side castle---#
                self.board[move.end_row][7] = self.board[move.end_row][move.end_col - 1]
                self.board[move.end_row][move.end_col - 1] = ''
            else:  #---Queen-side castle---#
                self.board[move.end_row][0] = self.board[move.end_row][move.end_col + 1]
                self.board[move.end_row][move.end_col + 1] = ''

        #---gives turn back---#
        self.white_to_move = not self.white_to_move

        #---Reset game end flags---#
        self.checkmate = False
        self.stalemate = False

    def get_valid_moves(self):
        moves = []

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == '':
                    continue

                if (piece.isupper() and self.white_to_move) or (piece.islower() and not self.white_to_move):
                    piece_type = piece.lower()

                    # Generate moves based on piece type
                    if piece_type == 'p':
                        self._get_pawn_moves(row, col, moves)
                    elif piece_type == 'r':
                        self._get_sliding_moves(row, col, [(-1, 0), (0, -1), (1, 0), (0, 1)], moves)
                    elif piece_type == 'n':
                        self._get_knight_moves(row, col, moves)
                    elif piece_type == 'b':
                        self._get_sliding_moves(row, col, [(-1, -1), (-1, 1), (1, -1), (1, 1)], moves)
                    elif piece_type == 'q':
                        self._get_sliding_moves(row, col, [(-1, 0), (0, -1), (1, 0), (0, 1),
                                                           (-1, -1), (-1, 1), (1, -1), (1, 1)], moves)
                    elif piece_type == 'k':
                        self._get_king_moves(row, col, moves)

        #---Filters out moves that would put or leave the king in check---#
        legal_moves = []
        for move in moves:
            self.make_move(move)
            self.white_to_move = not self.white_to_move  #---Switch back for check test---#
            in_check = self.in_check()
            self.white_to_move = not self.white_to_move
            self.undo_move()

            if not in_check:
                legal_moves.append(move)

        #---Check for checkmate or stalemate---#
        if len(legal_moves) == 0:
            if self.in_check():
                self.checkmate = True
            else:
                self.stalemate = True

        return legal_moves

    def in_check(self):
        if self.white_to_move:
            return self._square_under_attack(self.white_king_pos[0], self.white_king_pos[1])
        else:
            return self._square_under_attack(self.black_king_pos[0], self.black_king_pos[1])

    def _square_under_attack(self, row, col):
        self.white_to_move = not self.white_to_move
        enemy_moves = []

        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece != '' and (
                        (piece.isupper() and self.white_to_move) or (piece.islower() and not self.white_to_move)):
                    piece_type = piece.lower()

                    #---Generate attack patterns to check if they hit the square---#
                    if piece_type == 'p':
                        self._get_pawn_attacks(r, c, enemy_moves)
                    elif piece_type == 'r':
                        self._get_sliding_moves(r, c, [(-1, 0), (0, -1), (1, 0), (0, 1)], enemy_moves)
                    elif piece_type == 'n':
                        self._get_knight_moves(r, c, enemy_moves)
                    elif piece_type == 'b':
                        self._get_sliding_moves(r, c, [(-1, -1), (-1, 1), (1, -1), (1, 1)], enemy_moves)
                    elif piece_type == 'q':
                        self._get_sliding_moves(r, c, [(-1, 0), (0, -1), (1, 0), (0, 1),
                                                       (-1, -1), (-1, 1), (1, -1), (1, 1)], enemy_moves)
                    elif piece_type == 'k':
                        self._get_king_moves(r, c, enemy_moves, check_castle=False)

        self.white_to_move = not self.white_to_move  #---Switches turns back---#

        #---Check if any enemy move would land on the square---#
        for move in enemy_moves:
            if move.end_row == row and move.end_col == col:
                return True
        return False

    def _get_pawn_moves(self, row, col, moves):
        if self.white_to_move:  #---White pawn---#
            direction = -1
            start_row = 6
            enemy_pieces = lambda p: p != '' and p.islower()
        else:  #---Black pawn---#
            direction = 1
            start_row = 1
            enemy_pieces = lambda p: p != '' and p.isupper()

        #---Forward move---#
        if 0 <= row + direction < 8 and self.board[row + direction][col] == '':
            moves.append(Move((row, col), (row + direction, col), self))
            # Double move from starting position
            if row == start_row and self.board[row + 2 * direction][col] == '':
                moves.append(Move((row, col), (row + 2 * direction, col), self))

        #---Captures---#
        for c_offset in [-1, 1]:
            if 0 <= row + direction < 8 and 0 <= col + c_offset < 8:
                if enemy_pieces(self.board[row + direction][col + c_offset]):
                    moves.append(Move((row, col), (row + direction, col + c_offset), self))

    def _get_pawn_attacks(self, row, col, moves):
        if self.white_to_move:  #---White---#
            direction = -1
        else:  #---Black---#
            direction = 1

        #---Attack squares (diagonals)---#
        for c_offset in [-1, 1]:
            if 0 <= row + direction < 8 and 0 <= col + c_offset < 8:
                moves.append(Move((row, col), (row + direction, col + c_offset), self))

    def _get_sliding_moves(self, row, col, directions, moves):#---for rooks bishops and queens---#
        for d_row, d_col in directions:
            for i in range(1, 8):
                end_row = row + d_row * i
                end_col = col + d_col * i

                if not (0 <= end_row < 8 and 0 <= end_col < 8):
                    break

                end_piece = self.board[end_row][end_col]
                if end_piece == '':  # Empty square
                    moves.append(Move((row, col), (end_row, end_col), self))
                elif (end_piece.islower() and self.white_to_move) or (end_piece.isupper() and not self.white_to_move):

                    moves.append(Move((row, col), (end_row, end_col), self))
                    break
                else:
                    break

    def _get_knight_moves(self, row, col, moves):
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

        for m_row, m_col in knight_moves:
            end_row = row + m_row
            end_col = col + m_col

            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece == '' or (end_piece.islower() and self.white_to_move) or (
                        end_piece.isupper() and not self.white_to_move):
                    moves.append(Move((row, col), (end_row, end_col), self))

    def _get_king_moves(self, row, col, moves, check_castle=True):
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for m_row, m_col in king_moves:
            end_row = row + m_row
            end_col = col + m_col

            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece == '' or (end_piece.islower() and self.white_to_move) or (
                        end_piece.isupper() and not self.white_to_move):
                    moves.append(Move((row, col), (end_row, end_col), self))

        #---check castling---#
        if check_castle:
            self._get_castle_moves(row, col, moves)

    def _get_castle_moves(self, row, col, moves):
        if self.white_to_move:
            if self.castling_rights['K'] and self.board[7][5] == '' and self.board[7][6] == '':
                if not (self._square_under_attack(7, 4) or self._square_under_attack(7, 5) or self._square_under_attack(
                        7, 6)):
                    moves.append(Move((7, 4), (7, 6), self, is_castle=True))
            if self.castling_rights['Q'] and self.board[7][1] == '' and self.board[7][2] == '' and self.board[7][
                3] == '':
                if not (self._square_under_attack(7, 4) or self._square_under_attack(7, 3) or self._square_under_attack(
                        7, 2)):
                    moves.append(Move((7, 4), (7, 2), self, is_castle=True))
        else:
            if self.castling_rights['k'] and self.board[0][5] == '' and self.board[0][6] == '':
                if not (self._square_under_attack(0, 4) or self._square_under_attack(0, 5) or self._square_under_attack(
                        0, 6)):
                    moves.append(Move((0, 4), (0, 6), self, is_castle=True))
            if self.castling_rights['q'] and self.board[0][1] == '' and self.board[0][2] == '' and self.board[0][
                3] == '':
                if not (self._square_under_attack(0, 4) or self._square_under_attack(0, 3) or self._square_under_attack(
                        0, 2)):
                    moves.append(Move((0, 4), (0, 2), self, is_castle=True))


class ChessAI:
    def __init__(self, depth = 3):#---search depth is et to 3---#
        self.depth = depth

    def find_best_move(self, board):
        #---using minimax with alpha beta pruning---#
        self.nodes_searched = 0
        valid_moves = board.get_valid_moves()

        best_move = None
        best_score = float('-inf') if board.white_to_move else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in valid_moves:
            board.make_move(move)
            score = self.minimax(board, self.depth - 1, alpha, beta, not board.white_to_move)
            board.undo_move()

            if board.white_to_move and score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)
            elif not board.white_to_move and score < best_score:
                best_score = score
                best_move = move
                beta = min(beta, score)

        print(f"AI evaluated {self.nodes_searched} positions")
        return best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing):

        self.nodes_searched += 1

        if depth == 0:
            return self.evaluate_board(board)

        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:  #---checkmate or slatemate---#
            if board.in_check():
                return -10000 if is_maximizing else 10000
            else:
                return 0

        if is_maximizing:
            max_score = float('-inf')
            for move in valid_moves:
                board.make_move(move)
                score = self.minimax(board, depth - 1, alpha, beta, False)
                board.undo_move()
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = float('inf')
            for move in valid_moves:
                board.make_move(move)
                score = self.minimax(board, depth - 1, alpha, beta, True)
                board.undo_move()
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score

    def evaluate_board(self, board):

        if board.checkmate:
            return -10000 if board.white_to_move else 10000

        if board.stalemate:
            return 0

        score = 0

        for row in range(8):
            for col in range(8):
                piece = board.board[row][col]
                if piece in PIECE_VALUES:
                    score += PIECE_VALUES[piece]

                    #---bonus---#
                    if piece.lower() == 'p':  #---pawns---#
                        #---advance---#
                        if piece == 'P':
                            score += (7 - row) * 0.1
                        else:
                            score -= row * 0.1
                    elif piece.lower() in ['n', 'b']:  #---Knights and Bishops---#
                        #---Control center---#
                        center_dist = abs(3.5 - row) + abs(3.5 - col)
                        if piece.isupper():
                            score += (4 - center_dist) * 0.1
                        else:
                            score -= (4 - center_dist) * 0.1

        return score


def load_images():

    pieces = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']
    images = {}
    for piece in pieces:
        img_surface = pygame.Surface((SQ_SIZE - 10, SQ_SIZE - 10))
        color = WHITE if piece.isupper() else BLACK
        img_surface.fill(color)
        font = pygame.font.SysFont('Arial', 36, bold=True)
        text = font.render(piece, True, BLACK if piece.isupper() else WHITE)
        text_rect = text.get_rect(center=(img_surface.get_width() / 2, img_surface.get_height() / 2))
        img_surface.blit(text, text_rect)
        images[piece] = img_surface
    return images


class GameState:
    def __init__(self):
        self.board = Board()
        self.valid_moves = self.board.get_valid_moves()
        self.move_made = False
        self.human_white = True  #---human as default white---#
        self.game_over = False
        self.ai = ChessAI(depth = 3)

    def reset_game(self):
        self.board = Board()
        self.valid_moves = self.board.get_valid_moves()
        self.move_made = False
        self.game_over = False

    def switch_player(self):
        self.human_white = not self.human_white
        self.reset_game()

    def make_human_move(self, move):
        for valid_move in self.valid_moves:
            if move.start_row == valid_move.start_row and move.start_col == valid_move.start_col and \
                    move.end_row == valid_move.end_row and move.end_col == valid_move.end_col:
                self.board.make_move(valid_move)
                self.move_made = True
                return True
        return False

    def make_ai_move(self):
        move = self.ai.find_best_move(self.board)
        if move:
            self.board.make_move(move)
            self.move_made = True

    def update_valid_moves(self):#---updates each time a move is made---#
        if self.move_made:
            self.valid_moves = self.board.get_valid_moves()
            self.move_made = False

            #---checks if game is over----#
            if len(self.valid_moves) == 0:
                self.game_over = True


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CHESS BOT")
    clock = pygame.time.Clock()

    gs = GameState()
    images = load_images()
    running = True

    sq_selected = ()  # (row, col) of last click
    player_clicks = []  # Two tuples: [(row, col), (row, col)]

    while running:
        human_turn = (gs.board.white_to_move and gs.human_white) or (not gs.board.white_to_move and not gs.human_white)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            #---handlers---#
            elif event.type == pygame.MOUSEBUTTONDOWN and human_turn and not gs.game_over:
                location = pygame.mouse.get_pos()
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE

                if sq_selected == (row, col):  #---clicked same square twice---#
                    sq_selected = ()
                    player_clicks = []
                else:  #---different square---#
                    sq_selected = (row, col)
                    player_clicks.append(sq_selected)

                if len(player_clicks) == 2:  #---after second click---#
                    move = Move(player_clicks[0], player_clicks[1], gs.board)
                    if gs.make_human_move(move):
                        print("Human moved", player_clicks)

                    sq_selected = ()
                    player_clicks = []

            #---key handlers---#
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  #---reset---#
                    gs.reset_game()
                    sq_selected = ()
                    player_clicks = []
                elif event.key == pygame.K_s:  #---switch sides---#
                    gs.switch_player()
                    sq_selected = ()
                    player_clicks = []
                elif event.key == pygame.K_u and len(gs.board.move_log) > 0:  #---undo---#
                    gs.board.undo_move()
                    gs.move_made = True

        #---AI move---#
        if not gs.game_over and not human_turn:
            gs.make_ai_move()

        #---update game state---#
        gs.update_valid_moves()

        #---draws board---#
        draw_game_state(screen, gs, sq_selected, images)

        #---display game over message---#
        if gs.game_over:
            font = pygame.font.SysFont("Arial", 32, True)
            if gs.board.checkmate:
                text = "Black wins by checkmate!" if gs.board.white_to_move else "White wins by checkmate!"
            else:
                text = "Stalemate! Game is a draw."
            text_surface = font.render(text, True, BLACK)
            text_rect = text_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2))
            screen.blit(text_surface, text_rect)

        clock.tick(FPS)
        pygame.display.flip()


def draw_game_state(screen, gs, selected_sq, images):
    draw_board(screen)
    highlight_squares(screen, gs, selected_sq)
    draw_pieces(screen, gs.board.board, images)


def draw_board(screen):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def highlight_squares(screen, gs, selected_sq):
    if selected_sq:
        row, col = selected_sq

        #highlight selected square---#
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(HIGHLIGHT)
        screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

        #---highlight valid moves---#
        for move in gs.valid_moves:
            if move.start_row == row and move.start_col == col:
                s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                s.fill((124, 252, 0, 75))  #---light green---#
                screen.blit(s, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))


def draw_pieces(screen, board, images):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != '':
                screen.blit(images[piece], pygame.Rect(col * SQ_SIZE + 5, row * SQ_SIZE + 5, SQ_SIZE, SQ_SIZE))


if __name__ == "__main__":
    main()