"use client";

import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Loader2, Trash2, CheckCircle } from "lucide-react";
import { useState } from "react";
import { useQuery, useMutation } from "@apollo/client";
import { gql } from "@apollo/client";
import { toast } from "sonner";

// AIDEV-NOTE: 250903170012 - Todos page migrated from tRPC to GraphQL operations

// GraphQL operations for todos (example implementation)
const GET_TODOS = gql`
  query GetTodos {
    todos {
      id
      text
      completed
      createdAt
      updatedAt
    }
  }
`;

const CREATE_TODO = gql`
  mutation CreateTodo($input: CreateTodoInput!) {
    createTodo(input: $input) {
      id
      text
      completed
      createdAt
    }
  }
`;

const UPDATE_TODO = gql`
  mutation UpdateTodo($id: ID!, $input: UpdateTodoInput!) {
    updateTodo(id: $id, input: $input) {
      id
      text
      completed
      updatedAt
    }
  }
`;

const DELETE_TODO = gql`
  mutation DeleteTodo($id: ID!) {
    deleteTodo(id: $id) {
      success
      message
    }
  }
`;

export default function TodosPage() {
	const [newTodoText, setNewTodoText] = useState("");

	// GraphQL queries and mutations
	const { data, loading, error, refetch } = useQuery(GET_TODOS, {
		errorPolicy: 'all',
		pollInterval: 10000 // Refresh every 10 seconds
	});

	const [createTodo, { loading: createLoading }] = useMutation(CREATE_TODO, {
		onCompleted: () => {
			toast.success('Todo created successfully!');
			refetch();
			setNewTodoText("");
		},
		onError: (error) => {
			toast.error(`Failed to create todo: ${error.message}`);
		},
		refetchQueries: [{ query: GET_TODOS }]
	});

	const [updateTodo, { loading: updateLoading }] = useMutation(UPDATE_TODO, {
		onCompleted: () => {
			toast.success('Todo updated successfully!');
		},
		onError: (error) => {
			toast.error(`Failed to update todo: ${error.message}`);
		},
		refetchQueries: [{ query: GET_TODOS }]
	});

	const [deleteTodo, { loading: deleteLoading }] = useMutation(DELETE_TODO, {
		onCompleted: () => {
			toast.success('Todo deleted successfully!');
			refetch();
		},
		onError: (error) => {
			toast.error(`Failed to delete todo: ${error.message}`);
		},
		refetchQueries: [{ query: GET_TODOS }]
	});

	const handleAddTodo = async (e: React.FormEvent) => {
		e.preventDefault();
		if (newTodoText.trim()) {
			try {
				await createTodo({
					variables: {
						input: {
							text: newTodoText.trim()
						}
					}
				});
			} catch (error) {
				// Error handled by onError callback
			}
		}
	};

	const handleToggleTodo = async (id: string, completed: boolean) => {
		try {
			await updateTodo({
				variables: {
					id,
					input: {
						completed: !completed
					}
				}
			});
		} catch (error) {
			// Error handled by onError callback
		}
	};

	const handleDeleteTodo = async (id: string) => {
		if (window.confirm('Are you sure you want to delete this todo?')) {
			try {
				await deleteTodo({ variables: { id } });
			} catch (error) {
				// Error handled by onError callback
			}
		}
	};

	const todos = data?.todos || [];

	// Handle errors
	if (error) {
		return (
			<div className="mx-auto w-full max-w-md py-10">
				<Card>
					<CardHeader>
						<CardTitle className="text-destructive">Error Loading Todos</CardTitle>
						<CardDescription>{error.message}</CardDescription>
					</CardHeader>
					<CardContent>
						<Button onClick={() => refetch()} variant="outline" className="w-full">
							Try Again
						</Button>
					</CardContent>
				</Card>
			</div>
		);
	}

	return (
		<div className="mx-auto w-full max-w-md py-10">
			<Card>
				<CardHeader>
					<CardTitle className="flex items-center gap-2">
						<CheckCircle className="h-5 w-5" />
						Todo List
					</CardTitle>
					<CardDescription>Manage your tasks efficiently with GraphQL</CardDescription>
				</CardHeader>
				<CardContent>
					<form
						onSubmit={handleAddTodo}
						className="mb-6 flex items-center space-x-2"
					>
						<Input
							value={newTodoText}
							onChange={(e) => setNewTodoText(e.target.value)}
							placeholder="Add a new task..."
							disabled={createLoading}
							className="flex-1"
						/>
						<Button
							type="submit"
							disabled={createLoading || !newTodoText.trim()}
						>
							{createLoading ? (
								<Loader2 className="h-4 w-4 animate-spin" />
							) : (
								"Add"
							)}
						</Button>
					</form>

					{loading ? (
						<div className="flex justify-center py-4">
							<Loader2 className="h-6 w-6 animate-spin" />
							<span className="ml-2 text-sm text-muted-foreground">Loading todos...</span>
						</div>
					) : todos.length === 0 ? (
						<div className="py-8 text-center">
							<CheckCircle className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
							<p className="text-muted-foreground">No todos yet. Add one above!</p>
						</div>
					) : (
						<ul className="space-y-2">
							{todos.map((todo: any) => (
								<li
									key={todo.id}
									className="flex items-center justify-between rounded-md border p-3 transition-colors hover:bg-muted/50"
								>
									<div className="flex items-center space-x-3 flex-1">
										<Checkbox
											checked={todo.completed}
											onCheckedChange={() =>
												handleToggleTodo(todo.id, todo.completed)
											}
											id={`todo-${todo.id}`}
											disabled={updateLoading}
										/>
										<div className="flex-1">
											<label
												htmlFor={`todo-${todo.id}`}
												className={`cursor-pointer ${
													todo.completed 
														? "line-through text-muted-foreground" 
														: ""
												}`}
											>
												{todo.text}
											</label>
											{todo.createdAt && (
												<div className="text-xs text-muted-foreground mt-1">
													Created: {new Date(todo.createdAt).toLocaleDateString()}
												</div>
											)}
										</div>
									</div>
									<Button
										variant="ghost"
										size="icon"
										onClick={() => handleDeleteTodo(todo.id)}
										aria-label="Delete todo"
										disabled={deleteLoading}
										className="hover:text-destructive"
									>
										{deleteLoading ? (
											<Loader2 className="h-4 w-4 animate-spin" />
										) : (
											<Trash2 className="h-4 w-4" />
										)}
									</Button>
								</li>
							))}
						</ul>
					)}
					
					{todos.length > 0 && (
						<div className="mt-4 pt-4 border-t text-sm text-muted-foreground text-center">
							{todos.filter((todo: any) => !todo.completed).length} of {todos.length} tasks remaining
						</div>
					)}
				</CardContent>
			</Card>
		</div>
	);
}
