<script lang="ts">
	import { goto, beforeNavigate } from '$app/navigation';
	import TextInput from '$c/TextInput.svelte';
	import Button from '$c/Button.svelte';
	import FormGroup from '$c/FormGroup.svelte';
	import { auth, signup, clearErrors } from '$s/auth';
	import Alert from '$c/Alert.svelte';

	let email = '';
	let password = '';
	let passwordConfirm = '';

	function handleSubmit() {
		if (password !== passwordConfirm) {
			return alert('Passwords do not match');
		}
		signup(email, password);
	}

	$: if ($auth.user) {
		goto('/');
	}

	beforeNavigate(clearErrors);
</script>

<main class="w-full max-w-md mx-auto p-6">
	<div
		class="mt-7 bg-white border border-gray-200 rounded-xl shadow-sm dark:bg-gray-800 dark:border-gray-700"
	>
		<div class="p-4 sm:p-7">
			<div class="text-center">
				<h1 class="block text-2xl font-bold text-gray-800">Sign Up</h1>
				<p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
					Already have an account?
					<a class="text-blue-600 decoration-2 hover:underline font-medium" href="/auth/signin">
						Sign In Here
					</a>
				</p>
			</div>

			<div class="mt-5">
				<form on:submit|preventDefault={handleSubmit}>
					<div class="grid gap-y-4">
						<FormGroup label="Email">
							<TextInput bind:value={email} type="email" />
						</FormGroup>

						<FormGroup label="Password">
							<TextInput bind:value={password} type="password" />
						</FormGroup>

						<FormGroup label="Confirm Password">
							<TextInput bind:value={passwordConfirm} type="password" />
						</FormGroup>

						{#if $auth.error}
							<Alert>Error: {$auth.error}</Alert>
						{/if}

						<Button>Sign Up</Button>
					</div>
				</form>
			</div>
		</div>
	</div>
</main>
